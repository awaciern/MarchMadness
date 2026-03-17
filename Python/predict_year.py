"""
predict_year.py

Re-predict the bracket for a given year using an already-saved model.pkl and
overwrite the corresponding <YEAR>.html in its Predictions folder.

Unlike predict_brackets.py this script skips all training; it just loads the
pickled model and simulates the bracket deterministically (always picks the
team with the higher model win probability, identical to what predict_brackets
does when it runs with --this-year).

Usage
-----
    python3 Python/predict_year.py --model Predictions/GB_Best
    python3 Python/predict_year.py --model Predictions/GB_Best --year 2026
    python3 Python/predict_year.py --model Predictions/GB_Best \
        --year 2026 --final-four-pairings "0-2,1-3"
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket_html import format_bracket_html
from predict_brackets import (
    simulate_bracket,
    parse_ff_pairings_arg,
    derive_ff_pairings_from_data,
    TemperatureScaledModel,  # so pickle.load can deserialise calibrated models
)

try:
    from neural_net import TorchClassifier  # noqa: F401  needed for unpickling
except Exception:
    pass
try:
    from xgboost import XGBClassifier  # noqa: F401
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier  # noqa: F401
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(
        description='Re-predict a year\'s bracket from a saved model pickle.',
    )
    parser.add_argument(
        '--model', '-m', required=True,
        help='Path to a Predictions/<run-name> folder (or directly to model.pkl).',
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
    args = parser.parse_args()

    # Resolve paths ---------------------------------------------------------
    model_arg = Path(args.model)
    if model_arg.is_file():
        pkl_path = model_arg
        pred_dir = model_arg.parent
    else:
        pkl_path = model_arg / 'model.pkl'
        pred_dir = model_arg

    if not pkl_path.exists():
        print(f'ERROR: model pickle not found: {pkl_path}', file=sys.stderr)
        sys.exit(1)

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        # Assume this script lives in Python/ one level below the repo root.
        data_root = Path(__file__).resolve().parents[1]

    year = args.year

    # Load pkl --------------------------------------------------------------
    print(f'Loading model from {pkl_path} ...')
    with open(pkl_path, 'rb') as fh:
        payload = pickle.load(fh)

    model            = payload['model']
    model_key        = payload.get('model_key', 'unknown')
    feature_list     = payload['feature_list']
    model_feature_list = payload.get('model_feature_list', feature_list)
    cat_encoders     = payload.get('cat_encoders', {})
    norm_info        = payload.get('norm_info', None)
    delta_feats      = payload.get('delta_feats', False)
    numeric_bases    = payload.get('numeric_bases', [])
    pca_transformer  = payload.get('pca_transformer', None)

    # Load model_info for display metadata ----------------------------------
    info_path = pred_dir / 'model_info.json'
    if info_path.exists():
        with open(info_path) as fh:
            model_info = json.load(fh)
        feat_bases = model_info.get('feature_bases', [])
    else:
        feat_bases = []

    # Final Four pairings ---------------------------------------------------
    if args.final_four_pairings:
        ff_pairings = parse_ff_pairings_arg(args.final_four_pairings)
    else:
        # Try to derive from actual bracket data (works for completed years).
        try:
            ff_pairings = derive_ff_pairings_from_data(data_root, year)
            print(f'Derived FF pairings from data: {ff_pairings}')
        except Exception:
            ff_pairings = [(0, 1), (2, 3)]
            print(f'Using default FF pairings: {ff_pairings}')

    # Simulate bracket ------------------------------------------------------
    print(f'Predicting bracket for {year} ...')
    pred_teams, pred_seeds, pred_probs, correct, n_correct, score = simulate_bracket(
        model=model,
        data_root=data_root,
        year=year,
        this_year=year,          # treat it as the current year (no scoring)
        ff_pairings=ff_pairings,
        feature_list=feature_list,
        cat_encoders=cat_encoders,
        norm_info=norm_info,
        delta_feats=delta_feats,
        numeric_bases=numeric_bases,
        model_feature_list=model_feature_list,
        pca_transformer=pca_transformer,
    )

    # Write HTML ------------------------------------------------------------
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
    )
    out_path.write_text(html_str, encoding='utf-8')
    print(f'Bracket written to: {out_path}')


if __name__ == '__main__':
    main()
