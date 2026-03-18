"""
ensemble_loyo.py — Evaluate a soft-vote ensemble of two models using LOYO.

For each leave-one-year-out fold, both models are trained identically to
how predict_brackets.py trains them (Mixup2 sim data, PCA, etc.), but at
prediction time their probabilities are averaged before making a decision.

Usage:
    python3 tmp/ensemble_loyo.py \\
        --pkl1 Predictions/13g_svc_d2full_C015_mixup2_pca8/model.pkl \\
        --pkl2 PredictionsModelTourney5to7_Top/11b_svc_core_C0.2_mixup2_pca20/model.pkl

The script reads the training configuration from each PKL (features, sim data,
PCA components, model type, model_params) and reproduces the LOYO loop.

Output: per-year accuracy for each model and the ensemble, plus the mean.
"""

import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / 'Python'))

from predict_brackets import (
    load_combined_games,
    apply_label_encoders,
    apply_year_norm,
    apply_delta_transform,
    mirror_augment,
    fit_global_scaler_delta,
    build_and_train_model,
    ALL_YEARS,
)

EXCLUDE_YEARS = {2012, 2013, 2014}


def load_pkl(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_sim_data(sim_id: str) -> pd.DataFrame:
    """Load SimulatedData<sim_id>/All.csv."""
    p = BASE / 'Data' / f'SimulatedData{sim_id}' / 'All.csv'
    return pd.read_csv(p)


_SIM_PREFIXES = [
    # (substring to match in folder name, sim_data ID for load_sim_data)
    ('mixup_a05',  'Mixup_a05'),
    ('mxa05',      'Mixup_a05'),
    ('mixup_a30',  'Mixup_a30'),
    ('mxa30',      'Mixup_a30'),
    ('gmmteams',   'GMMTeams'),
    ('gmm',        'GMMTeams'),
    ('bootstrappairs', 'BootstrapPairs'),
    ('bootstrap',  'BootstrapPairs'),
    ('fn10',       'FN10'),
    ('mixup2',     'Mixup2'),
    ('mixup',      'Mixup2'),
]


def infer_sim_from_path(pkl_path: str) -> Optional[str]:
    """Try to infer sim_data ID from the model folder name (case-insensitive)."""
    name = Path(pkl_path).parent.name.lower()
    for fragment, sim_id in _SIM_PREFIXES:
        if fragment in name:
            return sim_id
    return None


def prepare_config(pkl: dict, sim_override: Optional[str] = None) -> dict:
    """Extract everything needed to replicate training from a PKL payload."""
    # PKL may not store sim_data; prefer explicit override, then PKL value, then None.
    sim_data = sim_override if sim_override is not None else pkl.get('sim_data')
    return {
        'feature_list':       pkl['feature_list'],
        'model_feature_list': pkl['model_feature_list'],
        'numeric_bases':      pkl['numeric_bases'],
        'cat_encoders':       pkl.get('cat_encoders', {}),
        'delta_feats':        pkl.get('delta_feats', False),
        'pca_n':              pkl['pca_transformer'].n_components_ if pkl.get('pca_transformer') else None,
        'model_key':          pkl['model_key'],
        'model_params':       {k: _coerce_param(v)
                               for k, v in (pkl.get('model_params') or {}).items()},
        'sim_data':           sim_data,
    }


def _is_numeric(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _coerce_param(v):
    """Convert string parameter values: try int first, then float, then bool, then leave as str."""
    if isinstance(v, (int, float, bool)):
        return v
    v_str = str(v)
    if v_str.lower() == 'true':
        return True
    if v_str.lower() == 'false':
        return False
    if v_str.lower() == 'none':
        return None
    try:
        return int(v_str)
    except (ValueError, TypeError):
        pass
    try:
        return float(v_str)
    except (ValueError, TypeError):
        pass
    return v_str


def build_loyo_model(cfg: dict, df_all: pd.DataFrame, df_sim: Optional[pd.DataFrame],
                     test_year: int, norm_info: dict):
    """
    Train one LOYO fold model for the given config.
    norm_info is pre-fitted globally (passed in, not fitted per fold).
    Returns (model, fold_pca, pc_cols).
    """
    feature_list       = cfg['feature_list']
    model_feature_list = cfg['model_feature_list']
    numeric_bases      = cfg['numeric_bases']
    cat_encoders       = cfg['cat_encoders']
    delta_feats        = cfg['delta_feats']
    pca_n              = cfg['pca_n']
    model_key          = cfg['model_key']
    model_params       = cfg['model_params']

    # --- training data (exclude test_year + excluded years) ---
    train_mask = ~df_all['Year'].isin({test_year} | EXCLUDE_YEARS)
    df_train   = df_all[train_mask].copy()

    # --- sim data (exclude same years) ---
    if df_sim is not None:
        sim_mask = ~df_sim['Year'].isin({test_year} | EXCLUDE_YEARS)
        df_sim_slice = df_sim[sim_mask].copy()
        # Keep only columns present in df_train
        shared_cols = [c for c in df_sim_slice.columns if c in df_train.columns]
        df_train = pd.concat([df_train, df_sim_slice[shared_cols]], ignore_index=True)

    # --- apply preprocessing (norm_info fitted globally on all real data) ---
    if cat_encoders:
        df_train = apply_label_encoders(df_train, cat_encoders)
    df_train = apply_year_norm(df_train, norm_info)
    if delta_feats and numeric_bases:
        df_train = apply_delta_transform(df_train, numeric_bases)
    df_train = mirror_augment(df_train, model_feature_list)

    X_tr = df_train[model_feature_list]
    y_tr = df_train['Win__1']

    # --- PCA ---
    fold_pca = None
    pc_cols  = None
    if pca_n:
        pc_cols  = [f'PC{i}' for i in range(pca_n)]
        fold_pca = PCA(n_components=pca_n, random_state=42)
        X_tr     = pd.DataFrame(fold_pca.fit_transform(X_tr), columns=pc_cols)
        y_tr     = y_tr.reset_index(drop=True)

    model = build_and_train_model(model_key, X_tr, y_tr, model_params)
    return model, fold_pca, pc_cols


def predict_proba_cfg(cfg: dict, df_test: pd.DataFrame,
                      norm_info: dict, fold_pca, pc_cols) -> np.ndarray:
    """Apply config's preprocessing to df_test and return predict_proba output."""
    feature_list       = cfg['feature_list']
    model_feature_list = cfg['model_feature_list']
    numeric_bases      = cfg['numeric_bases']
    cat_encoders       = cfg['cat_encoders']
    delta_feats        = cfg['delta_feats']

    df = df_test.copy()
    if cat_encoders:
        df = apply_label_encoders(df, cat_encoders)
    df = apply_year_norm(df, norm_info)
    if delta_feats and numeric_bases:
        df = apply_delta_transform(df, numeric_bases)
    X = df[model_feature_list]
    if fold_pca is not None:
        X = pd.DataFrame(fold_pca.transform(X), columns=pc_cols)
    return X


def run_ensemble(pkl_path1: str, pkl_path2: str, weights=(0.5, 0.5),
                 sim_override1: Optional[str] = None,
                 sim_override2: Optional[str] = None):
    pkl1 = load_pkl(pkl_path1)
    pkl2 = load_pkl(pkl_path2)

    # Resolve sim data: explicit override > path inference > PKL stored value
    sim1_id = sim_override1 if sim_override1 else infer_sim_from_path(pkl_path1)
    sim2_id = sim_override2 if sim_override2 else infer_sim_from_path(pkl_path2)

    cfg1 = prepare_config(pkl1, sim_override=sim1_id)
    cfg2 = prepare_config(pkl2, sim_override=sim2_id)

    name1 = Path(pkl_path1).parent.name
    name2 = Path(pkl_path2).parent.name

    print(f'\nModel 1: {name1}')
    print(f'  features={len(cfg1["model_feature_list"])}, sim={cfg1["sim_data"]}, pca={cfg1["pca_n"]}, model={cfg1["model_key"]}')
    print(f'Model 2: {name2}')
    print(f'  features={len(cfg2["model_feature_list"])}, sim={cfg2["sim_data"]}, pca={cfg2["pca_n"]}, model={cfg2["model_key"]}')
    print(f'Weights: {weights[0]:.2f} / {weights[1]:.2f}')

    # Load game data separately for each model (separate NaN drops preserve full dataset for each)
    df_raw = load_combined_games(BASE)
    df_all1 = df_raw.dropna(subset=[c for c in cfg1['feature_list'] if c in df_raw.columns]).copy()
    df_all2 = df_raw.dropna(subset=[c for c in cfg2['feature_list'] if c in df_raw.columns]).copy()
    print(f'\nLoaded data: model1 has {len(df_all1)} rows, model2 has {len(df_all2)} rows')

    # Load sim data for each config
    sim1 = load_sim_data(cfg1['sim_data']) if cfg1['sim_data'] else None  # type: Optional[pd.DataFrame]
    sim2 = load_sim_data(cfg2['sim_data']) if cfg2['sim_data'] else None  # type: Optional[pd.DataFrame]

    years = [y for y in ALL_YEARS if y not in EXCLUDE_YEARS]

    # --- Fit norm_info globally on all real data for each model (consistent with predict_brackets.py) ---
    df_for_norm1 = apply_label_encoders(df_all1, cfg1['cat_encoders']) if cfg1['cat_encoders'] else df_all1
    norm1_global = fit_global_scaler_delta(df_for_norm1, cfg1['numeric_bases'])
    df_for_norm2 = apply_label_encoders(df_all2, cfg2['cat_encoders']) if cfg2['cat_encoders'] else df_all2
    norm2_global = fit_global_scaler_delta(df_for_norm2, cfg2['numeric_bases'])

    acc1_list  = []
    acc2_list  = []
    acc_ens_list = []

    print(f'\n{"="*60}')
    print(f'{"Year":>4}  {"Model1":>7}  {"Model2":>7}  {"Ensemble":>9}')
    print(f'{"="*60}')

    for test_year in years:
        # --- test data ---
        # Both df_all1 and df_all2 have all 63 games per year (confirmed no NaN
        # in either model's features), so test set rows are in the same order.
        df_test1 = df_all1[df_all1['Year'] == test_year].copy().reset_index(drop=True)
        df_test2 = df_all2[df_all2['Year'] == test_year].copy().reset_index(drop=True)
        y_true = df_test1['Win__1'].values

        # --- train both models ---
        m1, pca1, pc1 = build_loyo_model(cfg1, df_all1, sim1, test_year, norm1_global)
        m2, pca2, pc2 = build_loyo_model(cfg2, df_all2, sim2, test_year, norm2_global)

        # --- predict on model-specific test data ---
        X1 = predict_proba_cfg(cfg1, df_test1, norm1_global, pca1, pc1)
        X2 = predict_proba_cfg(cfg2, df_test2, norm2_global, pca2, pc2)

        proba1 = m1.predict_proba(X1)  # shape (63, 2); col 1 = P(Win__1 = True)
        proba2 = m2.predict_proba(X2)

        # For models where predict() != argmax(predict_proba()), use predict() for
        # individual accuracy (to match predict_brackets LOYO metric).
        preds1 = m1.predict(X1).astype(bool).astype(int)
        preds2 = m2.predict(X2).astype(bool).astype(int)

        # DIAGNOSTIC: also check m1 accuracy using model.score directly
        y_arr1 = y_true.astype(int)
        debug_score1 = (preds1 == y_arr1).mean()
        # (Remove this diagnostic line after confirming correctness)

        # Soft-vote ensemble: average probabilities (both rows are same-ordered games)
        avg_proba = weights[0] * proba1[:, 1] + weights[1] * proba2[:, 1]
        preds_ens = (avg_proba > 0.5).astype(int)

        acc1  = (preds1 == y_true.astype(int)).mean()
        acc2  = (preds2 == y_true.astype(int)).mean()
        acc_e = (preds_ens == y_true.astype(int)).mean()

        acc1_list.append(acc1)
        acc2_list.append(acc2)
        acc_ens_list.append(acc_e)

        marker = ' *** BEST' if acc_e > max(acc1, acc2) + 0.001 else (
                 ' <-- WORSE' if acc_e < min(acc1, acc2) - 0.001 else '')
        print(f'{test_year:>4}  {acc1:>7.4f}  {acc2:>7.4f}  {acc_e:>9.4f}{marker}')

    avg1  = sum(acc1_list)  / len(acc1_list)
    avg2  = sum(acc2_list)  / len(acc2_list)
    avg_e = sum(acc_ens_list) / len(acc_ens_list)

    print(f'{"="*60}')
    print(f'{"AVG":>4}  {avg1:>7.4f}  {avg2:>7.4f}  {avg_e:>9.4f}')
    print(f'\nEnsemble {"IMPROVED" if avg_e > max(avg1,avg2)+0.0001 else ("same" if abs(avg_e - max(avg1,avg2)) < 0.0001 else "DEGRADED")} vs best individual: {max(avg1,avg2):.4f} -> {avg_e:.4f}')

    return avg1, avg2, avg_e


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl1', required=True)
    ap.add_argument('--pkl2', required=True)
    ap.add_argument('--w1', type=float, default=0.5, help='Weight for model 1 (default 0.5)')
    ap.add_argument('--w2', type=float, default=0.5, help='Weight for model 2 (default 0.5)')
    ap.add_argument('--sim1', default=None,
                    help='Sim data ID for model 1 (e.g. Mixup2). If omitted, inferred from path.')
    ap.add_argument('--sim2', default=None,
                    help='Sim data ID for model 2 (e.g. Mixup2). If omitted, inferred from path.')
    args = ap.parse_args()

    run_ensemble(args.pkl1, args.pkl2, weights=(args.w1, args.w2),
                 sim_override1=args.sim1, sim_override2=args.sim2)
