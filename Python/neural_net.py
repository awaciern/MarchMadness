"""
neural_net.py

PyTorch neural network models for tabular sports prediction data.
Provides two architectures exposed through a single scikit-learn compatible
wrapper (TorchClassifier):

  DeepMLP
      Straightforward feed-forward network with BatchNorm + Dropout in every
      hidden layer.  Good general-purpose baseline.

  TabResNet
      ResNet-style architecture designed for tabular data.  Every pair of
      hidden layers forms a residual block: the block output is added to a
      linear projection of the block input.  Better gradient flow in deep
      networks; often more robust than plain MLP.

Both architectures use:
  - Binary cross-entropy loss (single win-probability output).
  - Adam optimiser + CosineAnnealingLR schedule.
  - Internal 15 % validation split for early stopping.
  - Device: MPS (Apple Silicon) → CUDA → CPU; auto-detected.

Usage inside predict_brackets.py:

    model = TorchClassifier(arch='resnet', hidden_size=256, n_layers=6,
                            dropout=0.3, lr=1e-3, epochs=500)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)
    acc   = model.score(X_test, y_test)

Model params accepted by both TorchClassifier keys:
    arch           str    'mlp' | 'resnet' | 'transformer'  default 'resnet'
    hidden_size    int    width of each hidden layer         default 128
    n_layers       int    number of hidden layers            default 4
    dropout        float  per-layer dropout probability      default 0.3
    lr             float  initial learning rate              default 1e-3
    epochs         int    max training epochs                default 400
    batch_size     int    mini-batch size                    default 256
    weight_decay   float  Adam L2 regularisation             default 1e-4
    patience       int    early stopping patience (epochs)   default 40
    val_frac       float  fraction held out for val/early-stop default 0.15
    random_state   int    seed for reproducibility           default 42
    verbose        bool   print epoch-level loss             default False
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Architecture: Deep MLP
# ---------------------------------------------------------------------------

class _MLPBlock(nn.Module):
    """Linear → BatchNorm → ReLU → Dropout."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepMLP(nn.Module):
    """
    Feed-forward network with BatchNorm + Dropout.

    Architecture:
        Input BN → n_layers × MLPBlock(h, h) → Linear(h, 1)

    Output is a raw logit (no sigmoid); use BCEWithLogitsLoss for training
    and sigmoid for inference.
    """

    def __init__(self, in_dim: int, hidden_size: int = 128,
                 n_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_dim)
        layers: list = [_MLPBlock(in_dim, hidden_size, dropout)]
        for _ in range(n_layers - 1):
            layers.append(_MLPBlock(hidden_size, hidden_size, dropout))
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.hidden(x)
        return self.out(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Architecture: TabResNet
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """
    Residual block:  x  →  BN → ReLU → Linear → BN → ReLU → Linear  →  + proj(x)

    A linear projection is applied to x whenever in_dim ≠ out_dim.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.proj(x)


class TabResNet(nn.Module):
    """
    ResNet-style network for tabular data.

    Architecture:
        Linear(in, h) → n_blocks × ResBlock(h, h) → BN → ReLU → Linear(h, 1)

    One residual block = two linear layers with a skip connection.
    ``n_layers`` controls the total number of linear layers (≈ n_blocks * 2).
    """

    def __init__(self, in_dim: int, hidden_size: int = 128,
                 n_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        n_blocks = max(1, n_layers // 2)
        self.stem = nn.Linear(in_dim, hidden_size)
        self.blocks = nn.Sequential(
            *[_ResBlock(hidden_size, hidden_size, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Architecture: Lightweight Transformer (feature-wise attention)
# ---------------------------------------------------------------------------

class _FeatureAttentionBlock(nn.Module):
    """
    Single transformer-style block operating over features (treated as tokens).

    Each feature is embedded to `d_model` dims; multi-head self-attention is
    applied across the feature dimension; output is pooled back to a vector.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_feats, d_model)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TabTransformer(nn.Module):
    """
    Lightweight Transformer for tabular data.

    Each input feature is independently projected to `d_model` dimensions
    (treating features as sequence tokens).  Self-attention is applied across
    features, then CLS-token pooled and passed to an MLP head.
    Two residual transformer blocks are used; a separate MLP projection
    head converts the pooled representation to a win-probability logit.

    Parameters
    ----------
    in_dim      : number of input features
    hidden_size : embedding dimension d_model (default 64)
    n_layers    : number of transformer blocks (default 2)
    dropout     : dropout rate (default 0.2)
    """

    def __init__(self, in_dim: int, hidden_size: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        # Clamp to 2+ and ensure divisibility for multi-head attention
        d_model = max(32, hidden_size)
        n_heads = max(1, min(4, d_model // 16))
        # make d_model divisible by n_heads
        d_model = (d_model // n_heads) * n_heads

        self.d_model = d_model
        # Project each feature scalar → d_model vector
        self.feat_proj = nn.Linear(1, d_model)
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.Sequential(
            *[_FeatureAttentionBlock(d_model, n_heads, dropout)
              for _ in range(max(1, n_layers))]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim)
        batch = x.size(0)
        # expand each feature to (batch, 1, d_model)
        tokens = self.feat_proj(x.unsqueeze(-1))   # (B, F, d_model)
        cls = self.cls_token.expand(batch, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)    # (B, F+1, d_model)

        # Sequential can't take (batch, seq, d) directly; apply blocks manually
        out = tokens
        for block in self.blocks:
            out = block(out)

        # Use CLS token output
        cls_out = out[:, 0]  # (B, d_model)
        return self.head(cls_out).squeeze(-1)


# ---------------------------------------------------------------------------
# Scikit-learn compatible wrapper
# ---------------------------------------------------------------------------

_ARCH_MAP = {
    'mlp':         DeepMLP,
    'resnet':      TabResNet,
    'transformer': TabTransformer,
}


class TorchClassifier:
    """
    Scikit-learn compatible wrapper for the PyTorch tabular classifiers.

    Parameters
    ----------
    arch          : 'mlp' | 'resnet' | 'transformer'   (default 'resnet')
    hidden_size   : hidden layer / embedding width      (default 128)
    n_layers      : number of hidden layers             (default 4)
    dropout       : dropout probability                 (default 0.3)
    lr            : initial learning rate               (default 1e-3)
    epochs        : maximum training epochs             (default 400)
    batch_size    : mini-batch size                     (default 256)
    weight_decay  : Adam L2 regularisation              (default 1e-4)
    patience      : early stopping patience in epochs   (default 40)
    val_frac      : fraction of training data held out  (default 0.15)
    random_state  : RNG seed                            (default 42)
    verbose       : print per-epoch loss                (default False)
    """

    def __init__(
        self,
        arch: str = 'resnet',
        hidden_size: int = 128,
        n_layers: int = 4,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 400,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        patience: int = 40,
        val_frac: float = 0.15,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.arch = arch
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state
        self.verbose = verbose

        self.model_: Optional[nn.Module] = None
        self.device_: Optional[torch.device] = None
        self.classes_ = np.array([0, 1])
        self.n_features_in_: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_numpy(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(X).to(self.device_)

    def _make_net(self, in_dim: int) -> nn.Module:
        arch_cls = _ARCH_MAP.get(self.arch)
        if arch_cls is None:
            raise ValueError(
                f"Unknown arch '{self.arch}'. Choose from {list(_ARCH_MAP)}."
            )
        return arch_cls(
            in_dim=in_dim,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=np.float32)

        n_total = len(X_np)
        n_val   = max(1, int(n_total * self.val_frac))
        rng     = np.random.default_rng(self.random_state)
        idx     = rng.permutation(n_total)

        X_val_np,  y_val_np  = X_np[idx[:n_val]],  y_np[idx[:n_val]]
        X_tr_np,   y_tr_np   = X_np[idx[n_val:]],  y_np[idx[n_val:]]

        self.device_ = _get_device()
        self.n_features_in_ = X_np.shape[1]

        self.model_ = self._make_net(self.n_features_in_).to(self.device_)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        loss_fn = nn.BCEWithLogitsLoss()

        X_tr_t = self._to_tensor(X_tr_np)
        y_tr_t = self._to_tensor(y_tr_np)
        X_val_t = self._to_tensor(X_val_np)
        y_val_t = self._to_tensor(y_val_np)

        ds = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')
        best_state    = None
        no_improve    = 0

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss_fn(logits, yb).backward()
                optimizer.step()
            scheduler.step()

            # Validation loss for early stopping
            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_val_t)
                val_loss   = loss_fn(val_logits, y_val_t).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone()
                                 for k, v in self.model_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f'  Epoch {epoch+1:4d}/{self.epochs}  val_loss={val_loss:.4f}')

            if no_improve >= self.patience:
                if self.verbose:
                    print(f'  Early stop at epoch {epoch+1} '
                          f'(best val_loss={best_val_loss:.4f})')
                break

        # Restore best weights
        if best_state is not None:
            self.model_.load_state_dict(
                {k: v.to(self.device_) for k, v in best_state.items()}
            )
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError('Call fit() before predict_proba().')
        X_np = self._to_numpy(X)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(self._to_tensor(X_np)).cpu().numpy()
        prob1 = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return np.column_stack([1.0 - prob1, prob1])

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5)  # bool dtype — pandas .where() compatible

    def score(self, X, y) -> float:
        preds = self.predict(X)
        return np.mean(preds == np.asarray(y))

    # pickle support
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model_ is not None:
            state['_model_state_dict'] = {
                k: v.cpu() for k, v in self.model_.state_dict().items()
            }
            state['_model_arch_cfg'] = {
                'arch': self.arch,
                'hidden_size': self.hidden_size,
                'n_layers': self.n_layers,
                'dropout': self.dropout,
                'n_features_in': self.n_features_in_,
            }
        state['model_'] = None
        state['device_'] = None
        return state

    def __setstate__(self, state):
        cfg = state.pop('_model_arch_cfg', None)
        sd  = state.pop('_model_state_dict', None)
        self.__dict__.update(state)
        self.device_ = _get_device()
        if cfg is not None and sd is not None:
            arch_cls = _ARCH_MAP[cfg['arch']]
            self.model_ = arch_cls(
                in_dim=cfg['n_features_in'],
                hidden_size=cfg['hidden_size'],
                n_layers=cfg['n_layers'],
                dropout=cfg['dropout'],
            ).to(self.device_)
            self.model_.load_state_dict(
                {k: v.to(self.device_) for k, v in sd.items()}
            )
            self.model_.eval()


# ---------------------------------------------------------------------------
# Module-level sentinel so predict_brackets.py can detect TorchClassifier
# ---------------------------------------------------------------------------
TORCH_MODEL_KEYS = frozenset(['torch_mlp', 'torch_resnet', 'torch_transformer'])


def make_torch_classifier(model_key: str, **params) -> TorchClassifier:
    """
    Factory function called by predict_brackets.py's build_and_train_model.

    model_key → default arch:
        torch_mlp         → arch='mlp'
        torch_resnet      → arch='resnet'
        torch_transformer → arch='transformer'
    """
    arch_defaults = {
        'torch_mlp':         'mlp',
        'torch_resnet':      'resnet',
        'torch_transformer': 'transformer',
    }
    arch = params.pop('arch', arch_defaults.get(model_key, 'resnet'))
    return TorchClassifier(arch=arch, **params)
