"""
ensemble3_loyo.py — Evaluate 3-model soft-vote or majority-vote ensemble using LOYO.

Usage:
    python3 tmp/ensemble3_loyo.py \\
        --pkl1 Predictions/13g_lda_d2full_mixup2_pca10/model.pkl \\
        --pkl2 PredictionsModelTourney5to7_Top/8i_lr_core_mixup2_pca20_c08/model.pkl \\
        --pkl3 PredictionsModelTourney5to7_Top/11b_svc_core_C0.2_mixup2_pca20/model.pkl \\
        --strategy soft|hard

Strategies:
  soft:  average predicted probabilities across all 3 models; threshold at 0.5
  hard:  majority vote on predict() outputs
  max:   each game: use the model with the highest |proba - 0.5| (most confident)
"""

import pickle
import sys
from pathlib import Path
from typing import Optional, List, Tuple

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

# ── sim data helpers ─────────────────────────────────────────────────────────

_SIM_PREFIXES = [
    ('mixup_a05',     'Mixup_a05'),
    ('mxa05',         'Mixup_a05'),
    ('mixup_a30',     'Mixup_a30'),
    ('mxa30',         'Mixup_a30'),
    ('gmmteams',      'GMMTeams'),
    ('gmm',           'GMMTeams'),
    ('bootstrappairs','BootstrapPairs'),
    ('bootstrap',     'BootstrapPairs'),
    ('fn10',          'FN10'),
    ('mixup2',        'Mixup2'),
    ('mixup',         'Mixup2'),
]


def infer_sim_from_path(pkl_path: str) -> Optional[str]:
    name = Path(pkl_path).parent.name.lower()
    for fragment, sim_id in _SIM_PREFIXES:
        if fragment in name:
            return sim_id
    return None


def load_pkl(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_sim_data(sim_id: str) -> pd.DataFrame:
    p = BASE / 'Data' / f'SimulatedData{sim_id}' / 'All.csv'
    return pd.read_csv(p)


def _coerce_param(v):
    if isinstance(v, (int, float, bool)):
        return v
    v_str = str(v)
    if v_str.lower() == 'true':  return True
    if v_str.lower() == 'false': return False
    if v_str.lower() == 'none':  return None
    try: return int(v_str)
    except (ValueError, TypeError): pass
    try: return float(v_str)
    except (ValueError, TypeError): pass
    return v_str


def prepare_config(pkl: dict, sim_override: Optional[str] = None) -> dict:
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


def build_loyo_model(cfg: dict, df_all: pd.DataFrame, df_sim: Optional[pd.DataFrame],
                     test_year: int, norm_info: dict):
    """Train one LOYO fold model. Returns (model, fold_pca, pc_cols)."""
    feature_list       = cfg['feature_list']
    model_feature_list = cfg['model_feature_list']
    numeric_bases      = cfg['numeric_bases']
    cat_encoders       = cfg['cat_encoders']
    delta_feats        = cfg['delta_feats']
    pca_n              = cfg['pca_n']
    model_key          = cfg['model_key']
    model_params       = cfg['model_params']

    train_mask = ~df_all['Year'].isin({test_year} | EXCLUDE_YEARS)
    df_train   = df_all[train_mask].copy()

    if df_sim is not None:
        sim_mask     = ~df_sim['Year'].isin({test_year} | EXCLUDE_YEARS)
        df_sim_slice = df_sim[sim_mask].copy()
        shared_cols  = [c for c in df_sim_slice.columns if c in df_train.columns]
        df_train     = pd.concat([df_train, df_sim_slice[shared_cols]], ignore_index=True)

    if cat_encoders:
        df_train = apply_label_encoders(df_train, cat_encoders)
    df_train = apply_year_norm(df_train, norm_info)
    if delta_feats and numeric_bases:
        df_train = apply_delta_transform(df_train, numeric_bases)
    df_train = mirror_augment(df_train, model_feature_list)

    X_tr = df_train[model_feature_list]
    y_tr = df_train['Win__1']

    fold_pca = None
    pc_cols  = None
    if pca_n:
        pc_cols  = [f'PC{i}' for i in range(pca_n)]
        fold_pca = PCA(n_components=pca_n, random_state=42)
        X_tr     = pd.DataFrame(fold_pca.fit_transform(X_tr), columns=pc_cols)
        y_tr     = y_tr.reset_index(drop=True)

    model = build_and_train_model(model_key, X_tr, y_tr, model_params)
    return model, fold_pca, pc_cols


def get_test_features(cfg: dict, df_test: pd.DataFrame, norm_info: dict,
                      fold_pca, pc_cols) -> np.ndarray:
    """Preprocess test data and return feature matrix."""
    df = df_test.copy()
    if cfg['cat_encoders']:
        df = apply_label_encoders(df, cfg['cat_encoders'])
    df = apply_year_norm(df, norm_info)
    if cfg['delta_feats'] and cfg['numeric_bases']:
        df = apply_delta_transform(df, cfg['numeric_bases'])
    X = df[cfg['model_feature_list']]
    if fold_pca is not None:
        X = pd.DataFrame(fold_pca.transform(X), columns=pc_cols)
    return X


def run_ensemble3(pkl_paths: List[str], strategy: str = 'soft',
                  sim_overrides: Optional[List[str]] = None,
                  weights: Optional[List[float]] = None):
    """
    Run a 3-model LOYO ensemble.

    pkl_paths   : list of up to 3 pickle paths
    strategy    : 'soft' (average proba), 'hard' (majority vote), 'max' (highest confidence per game)
    sim_overrides: list of sim IDs (or None entries for auto-infer)
    weights     : list of per-model weights for soft vote (default equal)
    """
    n = len(pkl_paths)
    if sim_overrides is None:
        sim_overrides = [None] * n
    if weights is None:
        weights = [1.0 / n] * n

    pkls  = [load_pkl(p) for p in pkl_paths]
    sims  = [(sim_overrides[i] or infer_sim_from_path(pkl_paths[i])) for i in range(n)]
    cfgs  = [prepare_config(pkls[i], sim_override=sims[i]) for i in range(n)]
    names = [Path(p).parent.name for p in pkl_paths]

    print(f'\n{"="*70}')
    print(f'3-Model Ensemble  strategy={strategy}')
    for i, (nm, cfg) in enumerate(zip(names, cfgs)):
        print(f'  Model {i+1}: {nm}  feats={len(cfg["model_feature_list"])} sim={cfg["sim_data"]} pca={cfg["pca_n"]} key={cfg["model_key"]}')
    print(f'Weights: {[round(w, 4) for w in weights]}')
    print(f'{"="*70}')

    # Load raw data
    df_raw = load_combined_games(BASE)

    df_alls = []
    for cfg in cfgs:
        df_all_i = df_raw.dropna(
            subset=[c for c in cfg['feature_list'] if c in df_raw.columns]).copy()
        df_alls.append(df_all_i)
    print(f'Data rows per model: {[len(d) for d in df_alls]}')

    # Load sim data
    df_sims = []
    for cfg in cfgs:
        ds = load_sim_data(cfg['sim_data']) if cfg['sim_data'] else None
        df_sims.append(ds)

    # Fit per-model global norm
    norms = []
    for df_all_i, cfg in zip(df_alls, cfgs):
        df_n = apply_label_encoders(df_all_i, cfg['cat_encoders']) if cfg['cat_encoders'] else df_all_i
        norms.append(fit_global_scaler_delta(df_n, cfg['numeric_bases']))

    years = [y for y in ALL_YEARS if y not in EXCLUDE_YEARS]

    acc_per_model = [[] for _ in range(n)]
    acc_ens_list  = []

    header = '  '.join(f'M{i+1:d}' for i in range(n))
    print(f'\n{"Year":>4}  {header:>{6*n}}  {"Ensemble":>9}')
    print('-' * (4 + 6*n + 2 + 10))

    for test_year in years:
        df_tests = []
        for df_all_i in df_alls:
            df_t = df_all_i[df_all_i['Year'] == test_year].copy().reset_index(drop=True)
            df_tests.append(df_t)

        y_true = df_tests[0]['Win__1'].values.astype(int)

        # Train all models
        models_pcas = []
        for i in range(n):
            m, pca, pc = build_loyo_model(cfgs[i], df_alls[i], df_sims[i],
                                          test_year, norms[i])
            models_pcas.append((m, pca, pc))

        # Predict
        Xs    = [get_test_features(cfgs[i], df_tests[i], norms[i],
                                   models_pcas[i][1], models_pcas[i][2])
                 for i in range(n)]
        probas = [models_pcas[i][0].predict_proba(Xs[i])[:, 1]
                  for i in range(n)]
        preds_each = [models_pcas[i][0].predict(Xs[i]).astype(bool).astype(int)
                      for i in range(n)]

        accs = [(p == y_true).mean() for p in preds_each]
        for i, a in enumerate(accs):
            acc_per_model[i].append(a)

        if strategy == 'soft':
            avg_proba = sum(weights[i] * probas[i] for i in range(n))
            preds_ens = (avg_proba > 0.5).astype(int)

        elif strategy == 'hard':
            # Majority vote on predict() outputs
            votes = np.stack(preds_each, axis=1)  # (63, n)
            preds_ens = (votes.sum(axis=1) * 2 > n).astype(int)  # majority

        elif strategy == 'max':
            # Per-game: pick the model most confident |proba - 0.5|
            conf = np.stack([np.abs(p - 0.5) for p in probas], axis=1)  # (63, n)
            best_idx = conf.argmax(axis=1)  # (63,)
            proba_best = np.array([probas[best_idx[g]][g] for g in range(len(y_true))])
            preds_ens = (proba_best > 0.5).astype(int)

        else:
            raise ValueError(f'Unknown strategy: {strategy}')

        acc_e = (preds_ens == y_true).mean()
        acc_ens_list.append(acc_e)

        best_m = max(accs)
        marker = ' ***' if acc_e > best_m + 0.001 else (' <<<' if acc_e < min(accs) - 0.001 else '')
        acc_str = '  '.join(f'{a:.4f}' for a in accs)
        print(f'{test_year:>4}  {acc_str}  {acc_e:>9.4f}{marker}')

    avgs   = [sum(acc_per_model[i]) / len(years) for i in range(n)]
    avg_e  = sum(acc_ens_list) / len(years)
    best_avg = max(avgs)

    print('-' * (4 + 6*n + 2 + 10))
    avg_str = '  '.join(f'{a:.4f}' for a in avgs)
    print(f'{"AVG":>4}  {avg_str}  {avg_e:>9.4f}')
    print()
    verdict = 'IMPROVED' if avg_e > best_avg + 0.0001 else ('same' if abs(avg_e - best_avg) < 0.0001 else 'DEGRADED')
    print(f'Ensemble {verdict} vs best individual: {best_avg:.4f} -> {avg_e:.4f}')
    if avg_e > 0.75:
        print(f'  *** BREAKTHROUGH: {avg_e:.4f} > 75% ***')

    return avgs, avg_e, years, acc_ens_list, acc_per_model


def _save_ensemble_results(
        run_name: str,
        pkl_paths: List[str],
        strategy: str,
        avgs: List[float],
        avg_e: float,
        years: List[int],
        acc_ens_list: List[float],
        weights=None,
) -> None:
    """Write Predictions/<run_name>/model_info.json and summary.txt for webapp discovery."""
    import json
    out_dir = BASE / 'Predictions' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [Path(p).parent.name for p in pkl_paths]
    n = len(names)

    info = {
        "run_name": run_name,
        "model_key": "ensemble",
        "score": 0,  # no bracket score — ensembles don't run predict_brackets
        "expert_tag": f"ENS{n}{strategy.upper()}",
        "features": '+'.join(names),
        "params": "",
        "norm_years": False,
        "norm_all": True,
        "calibrate": False,
        "calibrate_mode": None,
        "calibrate_target": None,
        "calibrate_temperature": None,
        "delta_feats": True,
        "exclude_years": [2012, 2013, 2014],
        "sim_data": None,
        "pca_components": None,
        "model_params": {"strategy": strategy, "n_models": n,
                         "weights": weights},
        "feature_bases": names,
        "trad_train_acc": None,
        "trad_test_acc": None,
        "loyo_avg_train_acc": None,
        "loyo_avg_test_acc": round(avg_e, 4),
    }
    (out_dir / 'model_info.json').write_text(json.dumps(info, indent=2))

    n_test_years = len(years)
    lines = [
        'LEAVE-ONE-YEAR-OUT ENSEMBLE PERFORMANCE',
        '',
        'Avg LOYO train acc : N/A (ensemble)',
        f'Avg LOYO test acc  : {avg_e:.4f}  ({n_test_years} years)',
        '',
        'Per-year ensemble accuracy:',
    ]
    for yr, acc in zip(years, acc_ens_list):
        lines.append(f'  {yr}: train=N/A   test={acc:.4f}')
    lines += [
        '',
        'Ensemble components:',
    ]
    for i, (nm, avg_i) in enumerate(zip(names, avgs)):
        lines.append(f'  Model {i + 1}: {nm}  (LOYO avg={avg_i:.4f})')
    if weights:
        lines.append(f'  Weights: {[round(w, 4) for w in weights]}')
    lines += [
        f'  Strategy: {strategy}',
        '',
        'TRADITIONAL 67/33 TRAIN-TEST SPLIT MODEL (for reference)',
        f'  Train acc: {avg_e:.4f}  |  Test acc: {avg_e:.4f}',
    ]
    (out_dir / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print(f'\nSaved ensemble results to Predictions/{run_name}/')


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl1', required=True)
    ap.add_argument('--pkl2', required=True)
    ap.add_argument('--pkl3', default=None, help='Third model (optional)')
    ap.add_argument('--pkl4', default=None, help='Fourth model (optional)')
    ap.add_argument('--pkl5', default=None, help='Fifth model (optional)')
    ap.add_argument('--strategy', default='soft', choices=['soft', 'hard', 'max'],
                    help='Ensemble strategy: soft/hard/max (default: soft)')
    ap.add_argument('--sim1', default=None)
    ap.add_argument('--sim2', default=None)
    ap.add_argument('--sim3', default=None)
    ap.add_argument('--sim4', default=None)
    ap.add_argument('--sim5', default=None)
    ap.add_argument('--w1', type=float, default=None)
    ap.add_argument('--w2', type=float, default=None)
    ap.add_argument('--w3', type=float, default=None)
    ap.add_argument('--w4', type=float, default=None)
    ap.add_argument('--w5', type=float, default=None)
    ap.add_argument('--run-name', default=None,
                    help='Save results as Predictions/<run-name>/ for webapp visibility')
    args = ap.parse_args()

    pkl_paths = [args.pkl1, args.pkl2]
    sim_ovrd  = [args.sim1, args.sim2]
    wts_raw   = [args.w1, args.w2]

    for pkl_x, sim_x, w_x in [
        (args.pkl3, args.sim3, args.w3),
        (args.pkl4, args.sim4, args.w4),
        (args.pkl5, args.sim5, args.w5),
    ]:
        if pkl_x:
            pkl_paths.append(pkl_x)
            sim_ovrd.append(sim_x)
            wts_raw.append(w_x)

    # Build weights list (equal if not specified)
    n = len(pkl_paths)
    if all(w is None for w in wts_raw):
        weights = None  # equal
    else:
        total = sum(w or 1.0 for w in wts_raw)
        weights = [(w or 1.0) / total for w in wts_raw]

    avgs, avg_e, years, acc_ens_list, acc_per_model = run_ensemble3(
        pkl_paths, strategy=args.strategy, sim_overrides=sim_ovrd,
        weights=weights)

    if args.run_name:
        _save_ensemble_results(
            args.run_name, pkl_paths, args.strategy,
            avgs, avg_e, years, acc_ens_list, weights=weights)
