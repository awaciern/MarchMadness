"""
predict_year.py

Re-predict the bracket for a given year using an already-saved model.pkl (or
an ensemble folder with model_info.json) and overwrite the corresponding
<YEAR>.html in the Predictions folder.

Supports both single models and ensembles.  An ensemble folder has no
model.pkl; instead model_info.json lists the component models by name and the
vote strategy ('hard' or 'soft').

Usage
-----
    # Single model
    python3 Python/predict_year.py --model Predictions/GB_Best
    python3 Python/predict_year.py --model Predictions/GB_Best --year 2026

    # Ensemble
    python3 Python/predict_year.py \
        --model PredictionsModelTourney8_EnsembleTop/ens5_hgb_lda_svc_lr_svcd2
"""

import os as _os_env
for _omp_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    _os_env.environ.setdefault(_omp_var, '1')
del _os_env, _omp_var

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket_html import format_bracket_html
from predict_brackets import (
    ALL_YEARS,
    simulate_bracket,
    parse_ff_pairings_arg,
    derive_ff_pairings_from_data,
    TemperatureScaledModel,  # so pickle.load can deserialise calibrated models
    load_bracket_round,
    load_kenpom, attach_kenpom,
    load_barttorvik, attach_barttorvik,
    load_barttorvik_2week, attach_barttorvik_2week,
    load_barttorvik_hotness, attach_barttorvik_hotness,
    apply_label_encoders,
    apply_year_norm_single,
    apply_delta_transform,
)

try:
    from neural_net import TorchClassifier  # noqa: F401  needed for unpickling
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier  # noqa: F401
except Exception:
    pass


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def find_component_pkl(repo_root: Path, name: str) -> 'Path | None':
    """Search all Predictions* subdirs of repo_root for a model named `name`.

    Returns the path to model.pkl, or None if not found.  When multiple
    matches exist (same name under different Predictions* dirs) the most
    recently modified one is returned.
    """
    candidates = sorted(
        repo_root.glob(f'Predictions*/{name}/model.pkl'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _apply_temp_scale(avg_p1: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to a combined-probability array.

    T < 1 stretches probabilities away from 0.5 (more confident).
    T > 1 compresses them toward 0.5 (less confident).
    The predicted winner (p > 0.5) is never changed.
    """
    p = np.clip(avg_p1, 1e-9, 1.0 - 1e-9)
    logit = np.log(p) - np.log(1.0 - p)
    return 1.0 / (1.0 + np.exp(-logit / T))


def fit_ensemble_calibration_T(
    components: list,
    data_root: Path,
    strategy: str,
    weights: 'list | None',
    exclude_years: 'list | None' = None,
    p_target: float = 0.97,
    percentile: float = 98.0,
) -> float:
    """Fit a stretch calibration temperature T for a post-combination ensemble.

    Collects combined ensemble probabilities for all available historical
    bracket rounds, then returns T = percentile(|logit(avg_p1)|, percentile)
    / logit(p_target).  Mirrors fit_temperature_stretch() from predict_brackets
    but works on the averaged probability after all components are combined.
    """
    years = [y for y in ALL_YEARS if y not in (exclude_years or [])]
    all_logits: list = []
    for year in years:
        for rnd in range(1, 7):
            try:
                df = load_bracket_round(data_root, year, rnd)
                all_p1: list = []
                for comp in components:
                    df_c = df.copy()
                    if comp['cat_encoders']:
                        df_c = apply_label_encoders(df_c, comp['cat_encoders'])
                    if comp['norm_info'] is not None:
                        df_c = apply_year_norm_single(df_c, year, comp['norm_info'])
                    if comp['delta_feats'] and comp['numeric_bases']:
                        df_c = apply_delta_transform(df_c, comp['numeric_bases'])
                    X = df_c[comp['model_feature_list']]
                    if comp['pca_transformer'] is not None:
                        X = comp['pca_transformer'].transform(X)
                    proba = comp['model'].predict_proba(X)
                    all_p1.append(proba[:, 1])
                if not all_p1:
                    continue
                if strategy == 'soft':
                    w = np.array(weights, dtype=float) if weights else np.ones(len(components))
                    w = w / w.sum()
                    avg_p1 = np.average(np.array(all_p1, dtype=float), axis=0, weights=w)
                else:
                    avg_p1 = np.mean(np.array(all_p1, dtype=float), axis=0)
                p = np.clip(avg_p1, 1e-9, 1.0 - 1e-9)
                logits = np.log(p) - np.log(1.0 - p)
                all_logits.extend(logits.tolist())
            except Exception:
                continue
    if len(all_logits) < 5:
        return 1.0
    logits_arr = np.array(all_logits)
    L_observed = float(np.percentile(np.abs(logits_arr), percentile))
    if L_observed < 1e-6:
        return 1.0
    L_target = float(np.log(p_target / (1.0 - p_target)))
    T = L_observed / L_target
    print(f'  Calibration fit: L_{percentile:.0f}th={L_observed:.4f}  L_target={L_target:.4f}  T={T:.4f}')
    return float(np.clip(T, 0.05, 20.0))


def simulate_bracket_ensemble(
    components: list,
    strategy: str,
    weights: 'list | None',
    data_root: Path,
    year: int,
    this_year: int = None,
    ff_pairings: 'List[Tuple[int,int]] | None' = None,
    calibrate_temperature: 'float | None' = None,
) -> tuple:
    """Simulate a bracket using an ensemble of component models.

    Each element of *components* is a dict with keys:
        model, feature_list, model_feature_list, cat_encoders, norm_info,
        delta_feats, numeric_bases, pca_transformer

    strategy: 'hard' → majority vote; 'soft' → average predicted probabilities.
    weights:  optional list of floats (only used for soft voting).

    Returns the same 6-tuple as simulate_bracket:
        pred_teams_by_round, pred_seeds_by_round, pred_probs_by_round,
        correct_by_round, num_correct_by_round, total_score
    """
    is_current = (this_year is not None and year == this_year)
    if ff_pairings is None:
        ff_pairings = [(0, 1), (2, 3)]

    all_feat_lists = [c['feature_list'] for c in components]
    needs_bt    = any(f.startswith('BT__')    for fl in all_feat_lists for f in fl)
    needs_bt2w  = any(f.startswith('BT2W__')  for fl in all_feat_lists for f in fl)
    needs_bthot = any(f.startswith('BTHOT__') for fl in all_feat_lists for f in fl)

    pred_teams_by_round:  list = []
    pred_seeds_by_round:  list = []
    pred_probs_by_round:  list = []
    correct_by_round:     list = []
    num_correct_by_round: list = []
    winner_votes_by_round: list = []
    total_score = 0

    prev_pred_teams: list = []
    team_seed_map:   dict = {}

    for rnd in range(1, 7):
        if rnd == 1 or not is_current:
            df_round_base = load_bracket_round(data_root, year, rnd)

        if rnd == 1:
            for _, row in df_round_base.iterrows():
                team_seed_map[row['Team__1']] = row['Seed__1']
                team_seed_map[row['Team__2']] = row['Seed__2']

        if rnd > 1:
            actual_winners = (
                df_round_base['Winning_Team'].reset_index(drop=True)
                if not is_current else None
            )

            if rnd == 5:
                matchup_teams = [
                    (prev_pred_teams[ff_pairings[0][0]], prev_pred_teams[ff_pairings[0][1]]),
                    (prev_pred_teams[ff_pairings[1][0]], prev_pred_teams[ff_pairings[1][1]]),
                ]
            else:
                matchup_teams = [
                    (prev_pred_teams[i], prev_pred_teams[i + 1])
                    for i in range(0, len(prev_pred_teams), 2)
                ]

            df_matchups = pd.DataFrame(matchup_teams, columns=['Team__1', 'Team__2'])
            df_kp = load_kenpom(data_root, year)
            df_round_base = attach_kenpom(df_matchups, df_kp)
            if needs_bt:
                df_bt = load_barttorvik(data_root, year)
                df_round_base = attach_barttorvik(df_round_base, df_bt)
            if needs_bt2w:
                df_bt2w = load_barttorvik_2week(data_root, year)
                df_round_base = attach_barttorvik_2week(df_round_base, df_bt2w)
            if needs_bthot:
                df_hot = load_barttorvik_hotness(data_root, year)
                df_round_base = attach_barttorvik_hotness(df_round_base, df_hot)

            df_round_base['Seed__1'] = df_round_base['Team__1'].map(team_seed_map)
            df_round_base['Seed__2'] = df_round_base['Team__2'].map(team_seed_map)
            if not is_current:
                df_round_base['Winning_Team'] = actual_winners

        raw_seeds_1 = df_round_base['Seed__1'].copy() if 'Seed__1' in df_round_base.columns else None
        raw_seeds_2 = df_round_base['Seed__2'].copy() if 'Seed__2' in df_round_base.columns else None

        # Run each component model ----------------------------------------
        all_hard: list = []   # list of bool arrays (n_games,) per component
        all_p1:   list = []   # list of float arrays P(team1 wins), or empty
        has_proba = True

        for comp in components:
            df_c = df_round_base.copy()
            if comp['cat_encoders']:
                df_c = apply_label_encoders(df_c, comp['cat_encoders'])
            if comp['norm_info'] is not None:
                df_c = apply_year_norm_single(df_c, year, comp['norm_info'])
            if comp['delta_feats'] and comp['numeric_bases']:
                df_c = apply_delta_transform(df_c, comp['numeric_bases'])
            X = df_c[comp['model_feature_list']]
            if comp['pca_transformer'] is not None:
                X = comp['pca_transformer'].transform(X)
            preds = comp['model'].predict(X).astype(bool)
            all_hard.append(preds)
            try:
                proba = comp['model'].predict_proba(X)
                all_p1.append(proba[:, 1])
            except AttributeError:
                has_proba = False

        # Combine ----------------------------------------------------------
        hard_arr = np.array(all_hard, dtype=float)   # (n_comp, n_games)
        n_comp = len(components)

        if strategy == 'soft' and has_proba and all_p1:
            w = np.array(weights, dtype=float) if weights else np.ones(n_comp)
            w = w / w.sum()
            avg_p1 = np.average(np.array(all_p1, dtype=float), axis=0, weights=w)
            final_preds = avg_p1 >= 0.5
            if calibrate_temperature is not None and calibrate_temperature != 1.0:
                avg_p1 = _apply_temp_scale(avg_p1, calibrate_temperature)
            win_probs = [float(max(p, 1 - p)) for p in avg_p1]
            winner_votes_by_round.append(None)  # no vote counts for soft strategy
        else:
            votes = hard_arr.sum(axis=0)
            if n_comp % 2 == 0:
                # tie-break: first model decides
                final_preds = np.where(
                    votes > n_comp / 2, True,
                    np.where(votes < n_comp / 2, False, all_hard[0])
                )
            else:
                final_preds = votes > (n_comp / 2)
            final_preds = np.array(final_preds, dtype=bool)
            # Track winner vote counts for HTML display
            winner_votes_by_round.append([
                int(votes[k]) if bool(final_preds[k]) else int(n_comp - votes[k])
                for k in range(len(final_preds))
            ])
            # Use average component probability when available; else fall back to vote fraction
            if has_proba and all_p1:
                avg_p1 = np.mean(np.array(all_p1, dtype=float), axis=0)
                if calibrate_temperature is not None and calibrate_temperature != 1.0:
                    avg_p1 = _apply_temp_scale(avg_p1, calibrate_temperature)
                win_probs = [
                    float(avg_p1[k]) if bool(final_preds[k]) else float(1 - avg_p1[k])
                    for k in range(len(final_preds))
                ]
            else:
                win_probs = [
                    float(votes[k] / n_comp) if bool(final_preds[k]) else float(1 - votes[k] / n_comp)
                    for k in range(len(final_preds))
                ]

        # Build output series ----------------------------------------------
        df_r = df_round_base.copy()
        df_r['Pred_Win__1'] = final_preds
        pred_teams = df_r['Team__1'].where(pd.Series(final_preds, index=df_r.index), df_r['Team__2'])
        s1 = raw_seeds_1 if raw_seeds_1 is not None else df_r['Seed__1']
        s2 = raw_seeds_2 if raw_seeds_2 is not None else df_r['Seed__2']
        pred_seeds = s1.where(pd.Series(final_preds, index=df_r.index), s2)

        pred_teams_by_round.append(pred_teams.tolist())
        pred_seeds_by_round.append(pred_seeds.tolist())
        pred_probs_by_round.append(win_probs)
        prev_pred_teams = pred_teams.tolist()

        def _prob_str(p):
            return f'{p:.0%}' if p is not None else ''

        if not is_current:
            correct = (pred_teams == df_r['Winning_Team']).tolist()
            n_correct = sum(correct)
            round_score = n_correct * (2 ** (rnd - 1)) * 10
            correct_by_round.append(correct)
            num_correct_by_round.append(n_correct)
            total_score += round_score
            picks_str = '  '.join(
                f'[{pred_seeds.iloc[k]}]{pred_teams.iloc[k]} {_prob_str(win_probs[k])} '
                f'{"✓" if correct[k] else "✗"}'
                for k in range(len(pred_teams))
            )
            print(f'  Round {rnd} ({n_correct} correct, {round_score} pts): {picks_str}')
        else:
            correct_by_round.append([])
            num_correct_by_round.append(0)
            picks_str = '  '.join(
                f'[{pred_seeds.iloc[k]}]{pred_teams.iloc[k]} {_prob_str(win_probs[k])}'
                for k in range(len(pred_teams))
            )
            print(f'  Round {rnd}: {picks_str}')

    return (pred_teams_by_round, pred_seeds_by_round, pred_probs_by_round,
            correct_by_round, num_correct_by_round, total_score,
            winner_votes_by_round)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _shared_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Re-predict a year\'s bracket from a saved model (single or ensemble).',
    )
    parser.add_argument(
        '--model', '-m', required=True,
        help='Path to a Predictions folder (single model) or an ensemble folder.',
    )
    parser.add_argument(
        '--year', '-y', type=int, default=2026,
        help='Year to predict (default: 2026).',
    )
    parser.add_argument(
        '--final-four-pairings', default=None,
        metavar='"i-j,k-l"',
        help=(
            'How the 4 predicted Elite Eight winners (indexed 0-3) are paired '
            'for the Final Four.  Format: "i-j,k-l", e.g. "0-2,1-3".  '
            'Default: "0-1,2-3".'
        ),
    )
    parser.add_argument(
        '--data-root', default=None,
        help='Path to repo root (contains Data/).  Auto-detected if omitted.',
    )
    return parser


def _resolve_args(args):
    """Return (pred_dir, data_root, year, ff_pairings)."""
    model_arg = Path(args.model)
    # pred_dir is always the directory, even if a .pkl was passed directly.
    pred_dir = model_arg.parent if model_arg.is_file() else model_arg

    data_root = (
        Path(args.data_root)
        if args.data_root
        else Path(__file__).resolve().parents[1]
    )
    year = args.year

    if args.final_four_pairings:
        ff_pairings = parse_ff_pairings_arg(args.final_four_pairings)
    else:
        try:
            ff_pairings = derive_ff_pairings_from_data(data_root, year)
            print(f'Derived FF pairings from data: {ff_pairings}')
        except Exception:
            ff_pairings = [(0, 1), (2, 3)]
            print(f'Using default FF pairings: {ff_pairings}')

    return pred_dir, data_root, year, ff_pairings


def _write_html(pred_dir, data_root, year, ff_pairings, model_key, feat_bases,
                pred_teams, pred_seeds, pred_probs, correct, n_correct, score,
                votes_by_round=None, n_components=None):
    out_path = pred_dir / f'{year}.html'
    html_str = format_bracket_html(
        data_root=data_root,
        year=year,
        pred_teams_by_round=pred_teams,
        pred_seeds_by_round=pred_seeds,
        pred_probs_by_round=pred_probs,
        correct_by_round=correct,
        num_correct_by_round=n_correct,
        total_score=score,
        is_current=True,
        model_key=model_key,
        feat_bases=feat_bases,
        ff_pairings=ff_pairings,
        votes_by_round=votes_by_round,
        n_components=n_components,
    )
    out_path.write_text(html_str, encoding='utf-8')
    print(f'Bracket written to: {out_path}')


def main():
    parser = _shared_args()
    args   = parser.parse_args()

    pred_dir, data_root, year, ff_pairings = _resolve_args(args)

    # Read model_info to determine type (single vs ensemble) ----------------
    info_path = pred_dir / 'model_info.json'
    model_info: dict = {}
    if info_path.exists():
        with open(info_path) as fh:
            model_info = json.load(fh)

    is_ensemble = model_info.get('model_key') == 'ensemble'

    if is_ensemble:
        _run_ensemble(pred_dir, model_info, data_root, year, ff_pairings)
    else:
        # Locate the pkl
        model_arg = Path(args.model)
        if model_arg.is_file():
            pkl_path = model_arg
        else:
            pkl_path = pred_dir / 'model.pkl'
        if not pkl_path.exists():
            print(f'ERROR: model pickle not found: {pkl_path}', file=sys.stderr)
            sys.exit(1)
        _run_single(pred_dir, pkl_path, model_info, data_root, year, ff_pairings)


def _run_single(pred_dir, pkl_path, model_info, data_root, year, ff_pairings):
    """Predict bracket for a single pickled model."""
    print(f'Loading model from {pkl_path} ...')
    with open(pkl_path, 'rb') as fh:
        payload = pickle.load(fh)

    model              = payload['model']
    model_key          = payload.get('model_key', 'unknown')
    feature_list       = payload['feature_list']
    model_feature_list = payload.get('model_feature_list', feature_list)
    cat_encoders       = payload.get('cat_encoders', {})
    norm_info          = payload.get('norm_info', None)
    delta_feats        = payload.get('delta_feats', False)
    numeric_bases      = payload.get('numeric_bases', [])
    pca_transformer    = payload.get('pca_transformer', None)

    feat_bases = model_info.get('feature_bases', [])

    print(f'Predicting bracket for {year} (single model: {model_key}) ...')
    pred_teams, pred_seeds, pred_probs, correct, n_correct, score = simulate_bracket(
        model=model,
        data_root=data_root,
        year=year,
        this_year=year,
        ff_pairings=ff_pairings,
        feature_list=feature_list,
        cat_encoders=cat_encoders,
        norm_info=norm_info,
        delta_feats=delta_feats,
        numeric_bases=numeric_bases,
        model_feature_list=model_feature_list,
        pca_transformer=pca_transformer,
    )
    _write_html(pred_dir, data_root, year, ff_pairings, model_key, feat_bases,
                pred_teams, pred_seeds, pred_probs, correct, n_correct, score)


def _run_ensemble(pred_dir, model_info, data_root, year, ff_pairings):
    """Predict bracket for an ensemble by loading all component model PKLs."""
    strategy         = model_info.get('model_params', {}).get('strategy', 'hard')
    weights          = model_info.get('model_params', {}).get('weights', None)
    component_names  = model_info.get('feature_bases', [])

    if not component_names:
        print('ERROR: ensemble model_info has no feature_bases (component list).', file=sys.stderr)
        sys.exit(1)

    print(f'Ensemble: {len(component_names)} components, strategy={strategy}')
    components = []
    for name in component_names:
        pkl_path = find_component_pkl(data_root, name)
        if pkl_path is None:
            print(f'ERROR: Cannot find model.pkl for component: {name}', file=sys.stderr)
            sys.exit(1)
        print(f'  Loading component: {name}  ({pkl_path})')
        with open(pkl_path, 'rb') as fh:
            payload = pickle.load(fh)
        fl = payload['feature_list']
        components.append({
            'model':              payload['model'],
            'feature_list':       fl,
            'model_feature_list': payload.get('model_feature_list', fl),
            'cat_encoders':       payload.get('cat_encoders', {}),
            'norm_info':          payload.get('norm_info', None),
            'delta_feats':        payload.get('delta_feats', False),
            'numeric_bases':      payload.get('numeric_bases', []),
            'pca_transformer':    payload.get('pca_transformer', None),
        })

    feat_bases = component_names

    calibrate_T = model_info.get('ensemble_calibrate_temperature')
    print(f'Predicting bracket for {year} (ensemble/{strategy}) ...')
    if calibrate_T is not None:
        print(f'  Calibration: T={calibrate_T:.4f}  (post-combination temperature scaling)')
    pred_teams, pred_seeds, pred_probs, correct, n_correct, score, winner_votes = simulate_bracket_ensemble(
        components=components,
        strategy=strategy,
        weights=weights,
        data_root=data_root,
        year=year,
        this_year=year,
        ff_pairings=ff_pairings,
        calibrate_temperature=calibrate_T,
    )
    _write_html(pred_dir, data_root, year, ff_pairings, 'ensemble', feat_bases,
                pred_teams, pred_seeds, pred_probs, correct, n_correct, score,
                votes_by_round=winner_votes, n_components=len(components))


if __name__ == '__main__':
    main()
