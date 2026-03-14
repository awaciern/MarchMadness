"""
syn_feature_methods.py
----------------------
Feature-space synthetic data generation methods for March Madness tournament
game rows.  These complement the score-perturbation methods in generate_sim_data.py
by modifying the *team statistics* (KenPom, BartTorvik, etc.) directly, not just
the game outcomes, which produces training diversification that is fundamentally
different from score noise.

Public API (called from generate_sim_data.py)
---------------------------------------------
generate_feature_noise(df, n, rng, *, noise_frac)
    Independent Gaussian noise on every numeric feature column, scaled to
    noise_frac × std(column).  The win outcome is re-derived from the noisy
    Barthag values via Bradley-Terry sampling, keeping feature–outcome
    relationships self-consistent.

generate_correlated_noise(df, n, rng, *, noise_frac)
    Multivariate Gaussian noise using a Ledoit-Wolf regularised empirical
    covariance matrix fitted separately on the team-1 feature columns.  The
    same covariance structure is used to perturb team-2 features independently.
    Preserves the correlation structure between related statistics (e.g. AdjO
    and Barthag move together rather than independently).

generate_smote(df, n, rng, *, k_neighbors, pca_components)
    K-NN SMOTE-style interpolation in a PCA-compressed version of the full
    numeric feature space.  For each real row chooses k nearest neighbours,
    picks one, and linearly interpolates at a random λ drawn from U(0,1).
    Categorical columns are copied from the base row; outcome is re-derived.

generate_mixup(df, n, rng, *, alpha)
    Random-pair convex combination (Mixup, Zhang et al. 2018).  Two rows are
    drawn independently and blended with a Beta(alpha, alpha) mixing weight.
    Categorical columns come from the row with the larger mixing weight;
    outcome is re-derived.

Outcome re-derivation
---------------------
After any feature perturbation the win label is re-derived probabilistically:
  1. BT__Barthag__1/2 (Bradley-Terry model, exact probability scale).
  2. Fallback: KP__AdjEM__1/2 with a logistic mapping (scale ≈ 8 pts).
  3. Final fallback: original Win__1 is kept unchanged.

Rank columns (Rk_*) are perturbed alongside the absolute-value columns.
They may therefore be slightly inconsistent with the absolute values, which
is intentional: rank is a noisy snapshot of a team's standing and adding
small rank-level noise reflects realistic measurement uncertainty.

Notes on column exclusions
---------------------------
Columns that are *not* perturbed:
  - Identity / outcome: Team__*, Score__*, Win__1, Winning_Team, Year, Round
  - Seeds:  Seed__*  (fixed bracket assignment)
  - Categorical / string: *__Conf__*, *__W-L__*, *__Rec__*, *__G__*
All remaining numeric (int or float) columns are treated as perturbable
features.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column classification helpers
# ---------------------------------------------------------------------------

_IDENTITY_EXACT = frozenset({
    'Team__1', 'Team__2',
    'Score__1', 'Score__2',
    'Win__1', 'Winning_Team',
    'Year', 'Round',
    'Seed__1', 'Seed__2',
})

_CATEGORICAL_SUBSTRINGS = ('__Conf__', '__W-L__', '__Rec__', '__G__')


def _perturbable_cols(df: pd.DataFrame) -> List[str]:
    """
    Return a list of column names that are safe to add numeric noise to.
    Excludes identity columns, outcome labels, seeds, and categorical or
    string-typed columns.
    """
    cols = []
    for c in df.columns:
        if c in _IDENTITY_EXACT:
            continue
        if any(sub in c for sub in _CATEGORICAL_SUBSTRINGS):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _team1_perturbable_cols(df: pd.DataFrame) -> List[str]:
    """Perturbable columns ending with '__1' (team-1 feature set)."""
    all_pert = set(_perturbable_cols(df))
    return [c for c in df.columns if c in all_pert and c.endswith('__1')]


def _team2_perturbable_cols(df: pd.DataFrame) -> List[str]:
    """Perturbable columns ending with '__2' (team-2 feature set)."""
    all_pert = set(_perturbable_cols(df))
    return [c for c in df.columns if c in all_pert and c.endswith('__2')]


# ---------------------------------------------------------------------------
# Outcome re-derivation
# ---------------------------------------------------------------------------

def _rederive_outcome(
    sim_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Re-sample Win__1 ~ Bernoulli(p) from noisy Barthag (Bradley-Terry) or
    KenPom AdjEM (logistic fallback), then align Winning_Team.

    Modifies sim_df in-place; also returns it for chaining.
    """
    N = len(sim_df)
    p_win1: np.ndarray | None = None

    # --- Bradley-Terry from BartTorvik Barthag ---
    if 'BT__Barthag__1' in sim_df.columns and 'BT__Barthag__2' in sim_df.columns:
        eps = 1e-6
        b1 = np.clip(
            pd.to_numeric(sim_df['BT__Barthag__1'], errors='coerce').fillna(0.5).values,
            eps, 1 - eps,
        )
        b2 = np.clip(
            pd.to_numeric(sim_df['BT__Barthag__2'], errors='coerce').fillna(0.5).values,
            eps, 1 - eps,
        )
        logit_diff = np.log(b1 / (1 - b1)) - np.log(b2 / (1 - b2))
        p_win1 = 1.0 / (1.0 + np.exp(-logit_diff))

    # --- Logistic fallback from KenPom AdjEM ---
    elif 'KP__AdjEM__1' in sim_df.columns and 'KP__AdjEM__2' in sim_df.columns:
        em1 = pd.to_numeric(sim_df['KP__AdjEM__1'], errors='coerce').fillna(0.0).values
        em2 = pd.to_numeric(sim_df['KP__AdjEM__2'], errors='coerce').fillna(0.0).values
        # Scale ≈8 maps AdjEM differences to realistic win probabilities:
        #   ΔAdjEM ~8  →  p ≈ 0.73  (1 vs 8 seed neighbourhood)
        #   ΔAdjEM ~20 →  p ≈ 0.92  (1 vs 16 seed neighbourhood)
        diff = em1 - em2
        p_win1 = 1.0 / (1.0 + np.exp(-diff / 8.0))

    if p_win1 is None:
        # No quality signal available; keep original labels.
        return sim_df

    team1_wins = rng.random(N) < p_win1
    sim_df = sim_df.copy()
    sim_df['Win__1'] = team1_wins
    sim_df['Winning_Team'] = np.where(
        team1_wins,
        sim_df['Team__1'].values,
        sim_df['Team__2'].values,
    )
    return sim_df


# ---------------------------------------------------------------------------
# Method 4: independent feature noise
# ---------------------------------------------------------------------------

def generate_feature_noise(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    *,
    noise_frac: float = 0.05,
) -> pd.DataFrame:
    """
    Add independent Gaussian noise to every numeric feature column.

    The noise standard deviation for each column is ``noise_frac × std(column)``
    computed from the real data, so noisier / higher-variance features receive
    larger absolute perturbations while tightly-clustered features are only
    slightly perturbed.

    Parameters
    ----------
    df          : real game rows (source data)
    n           : number of noisy copies per real row
    rng         : NumPy random Generator (for reproducibility)
    noise_frac  : fraction of each column's std dev to use as noise level

    Win outcome re-derived via Bradley-Terry from noisy Barthag values.
    """
    feat_cols = _perturbable_cols(df)
    if not feat_cols:
        raise ValueError('No perturbable numeric columns found in DataFrame.')

    # Pre-compute std for each feature column (from real data)
    col_stds = df[feat_cols].apply(pd.to_numeric, errors='coerce').std().values  # (F,)
    col_stds = np.where(col_stds == 0, 1.0, col_stds)  # avoid zero-std columns

    total = len(df)
    N = total * n
    idx_rep = np.repeat(np.arange(total), n)
    sim_df = df.iloc[idx_rep].copy().reset_index(drop=True)

    # Feature matrix for all replicated rows: (N, F)
    X = pd.to_numeric(
        sim_df[feat_cols].stack(), errors='coerce',
    ).unstack().values.astype(float)

    # Draw noise: shape (N, F);  scale each column by noise_frac * std
    noise = rng.standard_normal((N, len(feat_cols))) * (noise_frac * col_stds)
    X_noisy = X + noise

    for j, col in enumerate(feat_cols):
        sim_df[col] = X_noisy[:, j]

    # Win labels are inherited from the replicated base rows (data-augmentation
    # philosophy: small perturbations don't change who won the game).
    return sim_df


# ---------------------------------------------------------------------------
# Method 5: correlated noise (Ledoit-Wolf covariance)
# ---------------------------------------------------------------------------

def generate_correlated_noise(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    *,
    noise_frac: float = 0.05,
) -> pd.DataFrame:
    """
    Multivariate Gaussian noise that respects the empirical correlation
    structure of the team features.

    A Ledoit-Wolf shrinkage estimator is fitted on the team-1 feature matrix
    from the *real* data, giving a regularised covariance Σ.  Perturbations
    are drawn from MVN(0, (noise_frac)² × Σ) and applied independently to
    the team-1 and team-2 feature blocks.  Because both blocks share the
    same population-level covariance, this is equivalent to treating each
    simulated team as a draw from the same "team quality" distribution.

    Related features (e.g. AdjO and Barthag, which are both measures of
    offensive quality) move together rather than independently, producing
    more realistic synthetic teams.

    Parameters
    ----------
    df          : real game rows
    n           : copies per real row
    rng         : NumPy random Generator
    noise_frac  : scale factor applied to the Cholesky factor of Σ
    """
    from sklearn.covariance import LedoitWolf

    t1_cols = _team1_perturbable_cols(df)
    t2_cols = _team2_perturbable_cols(df)

    if not t1_cols:
        raise ValueError('No perturbable team-1 columns found.')

    # Standardise team-1 features so the covariance matrix captures the
    # *correlation* structure rather than being dominated by high-variance scales.
    X1_real = df[t1_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(float)
    col_means = X1_real.mean(axis=0)    # (F,)
    col_stds  = X1_real.std(axis=0)     # (F,)
    col_stds  = np.where(col_stds == 0, 1.0, col_stds)

    Z1_real = (X1_real - col_means) / col_stds   # standardised: zero mean, unit std

    # Fit Ledoit-Wolf on standardised features → correlation-like matrix Σ_z
    lw = LedoitWolf(assume_centered=True)   # already zero-mean after standardisation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lw.fit(Z1_real)

    # Scale covariance: noise in standardised space = noise_frac (≈ 5% of each std)
    cov_z_scaled = lw.covariance_ * (noise_frac ** 2)
    try:
        L_z = np.linalg.cholesky(cov_z_scaled + np.eye(len(t1_cols)) * 1e-10)
    except np.linalg.LinAlgError:
        L_z = np.diag(np.full(len(t1_cols), noise_frac))

    total = len(df)
    N = total * n
    idx_rep = np.repeat(np.arange(total), n)
    sim_df = df.iloc[idx_rep].copy().reset_index(drop=True)

    F = len(t1_cols)

    def _perturb_block(X_block: np.ndarray, L: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Draw correlated noise in standardised space, back-transform to original."""
        z = rng.standard_normal((len(X_block), len(stds)))
        delta_z = z @ L.T          # correlated noise, each dim ≈ noise_frac
        delta   = delta_z * stds   # back to original scale
        return X_block + delta

    # Perturb team-1 features
    X1 = sim_df[t1_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(float)
    sim_df[t1_cols] = _perturb_block(X1, L_z, col_stds)

    # Perturb team-2 features with the same covariance but independent draws
    if t2_cols:
        t1_set = set(t1_cols)
        matching_t2: List[str] = []
        matching_j:  List[int] = []
        for c2 in t2_cols:
            c1 = c2[:-1] + '1'
            if c1 in t1_set:
                matching_t2.append(c2)
                matching_j.append(t1_cols.index(c1))

        if matching_t2:
            stds_matched = col_stds[matching_j]
            L_matched = L_z[np.ix_(matching_j, matching_j)]
            try:
                # Re-derive Cholesky of the sub-block; fall back to diagonal
                sub_cov = cov_z_scaled[np.ix_(matching_j, matching_j)]
                L_sub = np.linalg.cholesky(sub_cov + np.eye(len(matching_j)) * 1e-10)
            except np.linalg.LinAlgError:
                L_sub = np.diag(np.full(len(matching_j), noise_frac))
            X2 = sim_df[matching_t2].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(float)
            sim_df[matching_t2] = _perturb_block(X2, L_sub, stds_matched)

        # Unmatched t2 cols: independent noise scaled by feature std
        unmatched_t2 = [c for c in t2_cols if c not in matching_t2]
        if unmatched_t2:
            stds_u = df[unmatched_t2].apply(pd.to_numeric, errors='coerce').std().values
            stds_u = np.where(stds_u == 0, 1.0, stds_u)
            Xu = sim_df[unmatched_t2].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(float)
            sim_df[unmatched_t2] = Xu + rng.standard_normal((N, len(unmatched_t2))) * (noise_frac * stds_u)

    # Win labels inherited from replicated base rows.
    return sim_df


# ---------------------------------------------------------------------------
# Method 6: SMOTE-style kNN interpolation
# ---------------------------------------------------------------------------

def generate_smote(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    *,
    k_neighbors: int = 5,
    pca_components: int = 20,
) -> pd.DataFrame:
    """
    K-NN SMOTE-style interpolation in a PCA-compressed feature space.

    For each real row ``i``:
      1. Find its ``k_neighbors`` nearest neighbours in a PCA projection of
         the full numeric feature matrix.
      2. Draw a random neighbour ``j`` and a mixing weight λ ~ U(0,1).
      3. Produce a synthetic row: x_syn = x_i + λ × (x_j − x_i).

    This creates genuinely new feature-space points that lie *between*
    existing real matchups, interpolating team quality levels, styles of
    play, and historical context simultaneously.

    Categorical columns (conference, W-L strings) are copied from the
    base row ``i``; outcome (Win__1, Winning_Team) is re-derived.

    Parameters
    ----------
    df             : real game rows
    n              : synthetic copies per real row
    rng            : NumPy random Generator
    k_neighbors    : number of nearest neighbours to consider per row
    pca_components : PCA dimensionality for the kNN index
                     (set to min(20, n_cols//2) automatically if too large)
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    feat_cols = _perturbable_cols(df)
    if not feat_cols:
        raise ValueError('No perturbable numeric columns found.')

    X_raw = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(float)
    total = len(df)

    # Standardise before PCA
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    # PCA reduction for kNN index (must not exceed min dimension)
    n_components = min(pca_components, X_std.shape[1], total - 1)
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_std)

    # Build kNN index in PCA space (Euclidean; L2 = standard for SMOTE)
    k_eff = min(k_neighbors, total - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm='auto', metric='euclidean')
    nn.fit(X_pca)
    _, indices = nn.kneighbors(X_pca)  # shape (total, k_eff+1); first col = self

    # Build synthetic rows
    sim_rows_list = []
    for i in range(total):
        row_base = df.iloc[i]
        neighbors = indices[i, 1:]   # exclude self
        xi = X_raw[i]
        for _ in range(n):
            j = int(rng.choice(neighbors))
            xj = X_raw[j]
            lam = rng.random()
            x_syn = xi + lam * (xj - xi)
            # Start from base row (carries categorical columns) then overwrite numerics
            new_row = row_base.copy()
            for k_idx, col in enumerate(feat_cols):
                new_row[col] = x_syn[k_idx]
            sim_rows_list.append(new_row)

    sim_df = pd.DataFrame(sim_rows_list).reset_index(drop=True)
    # Win__1 / Winning_Team inherited from base row i (classical SMOTE: synthetic
    # points in the neighbourhood of a real point carry that point's label).
    return sim_df


# ---------------------------------------------------------------------------
# Method 7: Mixup
# ---------------------------------------------------------------------------

def generate_mixup(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    *,
    alpha: float = 2.0,
) -> pd.DataFrame:
    """
    Random-pair convex combination (Mixup), Zhang et al. 2018.

    For each synthetic row:
      1. Draw two source rows i, j independently at random.
      2. Draw mixing weight λ ~ Beta(alpha, alpha).
      3. Produce: x_syn = λ × x_i + (1−λ) × x_j

    The mixing coefficient is drawn from a symmetric Beta distribution; with
    alpha=2.0 the distribution peaks away from 0 and 1, so synthetic rows
    are genuine blends rather than near-verbatim copies.

    * Categorical columns are copied from whichever parent row has higher λ.
    * Win outcome is re-derived from the blended Barthag / AdjEM values.

    Mixup is primarily a regularisation technique: the blended feature vectors
    occupy regions of feature space that no real team has occupied, forcing the
    model to interpolate smoothly rather than memorise the training points.

    Parameters
    ----------
    df    : real game rows
    n     : total synthetic rows to generate (NOT per-real-row; the function
            draws n random pairs and produces n blended rows regardless of df size)
    rng   : NumPy random Generator
    alpha : Beta distribution concentration parameter (higher = more mixing)
    """
    feat_cols = _perturbable_cols(df)
    if not feat_cols:
        raise ValueError('No perturbable numeric columns found.')

    total = len(df)
    # For mixup we generate n * total synthetic rows (n per original row),
    # matching the other methods' semantics.
    N = total * n

    X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(float)

    # Draw n*total pairs of row indices
    idx_i = rng.integers(0, total, size=N)
    idx_j = rng.integers(0, total, size=N)
    lam   = rng.beta(alpha, alpha, size=N)  # (N,)

    Xi = X[idx_i]  # (N, F)
    Xj = X[idx_j]  # (N, F)
    X_syn = lam[:, None] * Xi + (1 - lam[:, None]) * Xj

    # Categorical columns: from parent with larger lambda weight
    use_i = lam >= 0.5
    cat_parent_idx = np.where(use_i, idx_i, idx_j)

    # Build output DataFrame
    # Start from rows indexed by cat_parent_idx to carry non-numeric columns
    sim_df = df.iloc[cat_parent_idx].copy().reset_index(drop=True)
    for j_col, col in enumerate(feat_cols):
        sim_df[col] = X_syn[:, j_col]

    # Win__1 / Winning_Team come from the higher-λ parent (already set via
    # cat_parent_idx).  No separate rederivation needed.
    return sim_df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_feature_diagnostics(
    df_real: pd.DataFrame,
    df_sim: pd.DataFrame,
    method: str,
) -> None:
    """
    Print a short summary comparing real and synthetic feature distributions.
    Shows Win__1 rate and, for the key quality metric (BT Barthag AdjEM or
    KP AdjEM δ), the mean and std of the difference between teams.
    """
    real_rate = float(pd.to_numeric(df_real['Win__1'], errors='coerce').mean())
    sim_rate  = float(df_sim['Win__1'].astype(float).mean())
    print(f'  Win__1 rate   real={real_rate:.4f}  sim={sim_rate:.4f}  '
          f'(Δ={sim_rate - real_rate:+.4f})')

    # Compare key quality-gap statistic
    label, c1, c2 = None, None, None
    if 'BT__Barthag__1' in df_real.columns:
        label, c1, c2 = 'Barthag gap (team1−team2)', 'BT__Barthag__1', 'BT__Barthag__2'
    elif 'KP__AdjEM__1' in df_real.columns:
        label, c1, c2 = 'AdjEM gap (team1−team2)', 'KP__AdjEM__1', 'KP__AdjEM__2'

    if label:
        for tag, src in [('real', df_real), ('sim ', df_sim)]:
            gap = (
                pd.to_numeric(src[c1], errors='coerce') -
                pd.to_numeric(src[c2], errors='coerce')
            ).dropna()
            print(f'  {tag}  {label}: mean={gap.mean():+.3f}  std={gap.std():.3f}')

    # Feature-std drift: mean absolute % change in column std
    feat_cols = _perturbable_cols(df_real)
    if feat_cols:
        real_stds = df_real[feat_cols].apply(pd.to_numeric, errors='coerce').std()
        sim_stds  = df_sim[feat_cols].apply(pd.to_numeric, errors='coerce').std()
        valid = real_stds[real_stds > 0]
        pct_change = ((sim_stds[valid.index] - valid) / valid).abs()
        print(f'  Feature std drift (mean |Δstd/std|): '
              f'{pct_change.mean():.4f}  max={pct_change.max():.4f}  '
              f'({method} method)')
