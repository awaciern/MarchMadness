#!/usr/bin/env python3
"""
update_ensemble_summaries.py

Computes bracket stats for all ENS_* ensemble models in Predictions/ and:
  1. Regenerates HTML files with correct/incorrect markings for past years
  2. Updates summary.txt with per-year bracket scores and per-round accuracy

Run from the repo root:
    env/bin/python tmp/update_ensemble_summaries.py
"""

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np  # noqa: F401 – used by imported modules
import pandas as pd  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'Python'))

# Import at module level so pickle can deserialise calibrated models
try:
    from predict_brackets import TemperatureScaledModel  # noqa: F401
except Exception as e:
    print(f'WARNING: could not import TemperatureScaledModel: {e}')

try:
    from neural_net import TorchClassifier  # noqa: F401
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier  # noqa: F401
except Exception:
    pass

from bracket_html import format_bracket_html
from predict_year import simulate_bracket_ensemble, find_component_pkl
from predict_brackets import derive_ff_pairings_from_data

PREDICTIONS_DIR = REPO_ROOT / 'Predictions'
CURRENT_YEAR    = 2026   # no actual results yet

# All tournament years with known results (these get bracket scoring)
ALL_EVAL_YEARS = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                  2021, 2022, 2023, 2024, 2025]

# Sentinel used so is_current=False for all eval years
SENTINEL_YEAR = 9999


def load_components(model_info: dict) -> list:
    """Load all component PKLs for an ensemble model."""
    component_names = model_info.get('feature_bases', [])
    if not component_names:
        raise ValueError('model_info has no feature_bases (component list)')
    components = []
    for name in component_names:
        pkl_path = find_component_pkl(REPO_ROOT, name)
        if pkl_path is None:
            raise FileNotFoundError(f'Cannot find model.pkl for component: {name}')
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
    return components


def build_bracket_section(total_correct_by_round: dict, total_score: int,
                           num_eval_years: int) -> str:
    """Build the 'Bracket results:' text block."""
    games_per_round = [32, 16, 8, 4, 2, 1]
    lines = ['Bracket results:']
    for rnd in range(1, 7):
        total_games = games_per_round[rnd - 1] * num_eval_years
        n = total_correct_by_round[rnd]
        pct = n / total_games * 100 if total_games else 0.0
        pts = n * (2 ** (rnd - 1)) * 10
        lines.append(f'  Round {rnd}: {n}/{total_games} ({pct:.1f}%), {pts} pts')
    total_games_all = 63 * num_eval_years
    total_correct_all = sum(total_correct_by_round[r] for r in range(1, 7))
    if total_games_all:
        lines.append(
            f'  All rounds: {total_correct_all}/{total_games_all} '
            f'({total_correct_all / total_games_all * 100:.1f}%)'
        )
        avg = total_score / num_eval_years if num_eval_years else 0.0
        lines.append(f'  Avg bracket score: {avg:.1f}')
    return '\n'.join(lines)


def rebuild_summary(ens_dir: Path, model_info: dict, year_bracket_scores: dict,
                    total_correct_by_round: dict, total_score: int,
                    num_eval_years: int) -> None:
    """Rebuild summary.txt from scratch, preserving LOYO accuracy data and
    adding bracket stats.  Produces clean, consistent formatting every time."""
    summary_path = ens_dir / 'summary.txt'
    old_text = summary_path.read_text() if summary_path.exists() else ''

    # --- Extract per-year test accuracy from existing summary ---------------
    year_test_acc: dict = {}
    for m in re.finditer(
            r'^\s+(\d{4}): train=N/A\s+test=([\d.]+)', old_text, re.MULTILINE):
        year_test_acc[int(m.group(1))] = float(m.group(2))

    # --- Extract LOYO avg from existing summary -----------------------------
    loyo_m = re.search(
        r'Avg LOYO test acc\s+:\s+([\d.]+)\s+\((\d+) years\)', old_text)
    loyo_avg       = float(loyo_m.group(1)) if loyo_m else None
    num_loyo_years = int(loyo_m.group(2))   if loyo_m else len(year_test_acc)

    # --- Extract component descriptions -------------------------------------
    comp_lines = re.findall(r'^\s+(Model \d+: .+)$', old_text, re.MULTILINE)
    strat_m    = re.search(r'^\s+Strategy: (\S+)', old_text, re.MULTILINE)
    strategy   = strat_m.group(1) if strat_m else 'hard'

    # --- Extract TRADITIONAL section ----------------------------------------
    trad_m       = re.search(r'(TRADITIONAL 67/33 .+)', old_text, re.DOTALL)
    trad_section = trad_m.group(1).strip() if trad_m else ''

    # --- Build new summary --------------------------------------------------
    lines = ['LEAVE-ONE-YEAR-OUT ENSEMBLE PERFORMANCE', '']
    lines.append('Avg LOYO train acc : N/A (ensemble)')
    if loyo_avg is not None:
        lines.append(
            f'Avg LOYO test acc  : {loyo_avg:.4f}  ({num_loyo_years} years)')
    lines.append('')
    lines.append('Per-year ensemble accuracy:')
    for yr, test_acc in sorted(year_test_acc.items()):
        score = year_bracket_scores.get(yr)
        score_str = f'  bracket={score} pts' if score is not None else ''
        lines.append(f'  {yr}: train=N/A   test={test_acc:.4f}{score_str}')
    lines.append('')
    lines.append('Ensemble components:')
    for cl in comp_lines:
        lines.append(f'  {cl}')
    lines.append(f'  Strategy: {strategy}')
    lines.append('')
    lines.append(build_bracket_section(
        total_correct_by_round, total_score, num_eval_years))
    if trad_section:
        lines.append('')
        lines.append(trad_section)

    summary_path.write_text('\n'.join(lines))


def main():
    ens_dirs = sorted(
        d for d in PREDICTIONS_DIR.iterdir()
        if d.name.startswith('ENS_') and (d / 'model_info.json').exists()
    )

    if not ens_dirs:
        print('No ENS_* dirs found in Predictions/')
        return

    print(f'Found {len(ens_dirs)} ensemble models:')
    for d in ens_dirs:
        print(f'  {d.name}')

    for ens_dir in ens_dirs:
        print(f"\n{'='*60}")
        print(f'Processing: {ens_dir.name}')
        print(f"{'='*60}")

        with open(ens_dir / 'model_info.json') as fh:
            model_info = json.load(fh)

        strategy      = model_info.get('model_params', {}).get('strategy', 'hard')
        weights       = model_info.get('model_params', {}).get('weights', None)
        exclude_years = set(model_info.get('exclude_years', []))
        print(f'  Strategy: {strategy}  |  Exclude years: {sorted(exclude_years)}')

        eval_years = [y for y in ALL_EVAL_YEARS if y not in exclude_years]
        years_to_run = eval_years + [CURRENT_YEAR]
        print(f'  Eval years ({len(eval_years)}): {eval_years}')

        try:
            components = load_components(model_info)
        except (FileNotFoundError, ValueError) as e:
            print(f'  ERROR loading components: {e}')
            continue
        print(f'  Loaded {len(components)} component PKLs')

        year_bracket_scores: dict = {}
        total_correct_by_round = {r: 0 for r in range(1, 7)}
        total_score   = 0
        num_eval_years = 0

        for year in years_to_run:
            is_current = (year == CURRENT_YEAR)
            this_year  = CURRENT_YEAR if is_current else SENTINEL_YEAR

            try:
                ff_pairings = derive_ff_pairings_from_data(REPO_ROOT, year)
            except Exception:
                ff_pairings = [(0, 1), (2, 3)]

            print(f'  {year} (is_current={is_current}) ff={ff_pairings}  ... ',
                  end='', flush=True)
            try:
                (pred_teams, pred_seeds, pred_probs,
                 correct, n_correct, score,
                 winner_votes) = simulate_bracket_ensemble(
                    components=components,
                    strategy=strategy,
                    weights=weights,
                    data_root=REPO_ROOT,
                    year=year,
                    this_year=this_year,
                    ff_pairings=ff_pairings,
                )
            except Exception as e:
                print(f'ERROR: {e}')
                year_bracket_scores[year] = None
                continue

            if not is_current:
                year_bracket_scores[year] = score
                for rnd in range(1, 7):
                    total_correct_by_round[rnd] += n_correct[rnd - 1]
                total_score   += score
                num_eval_years += 1
                print(f'{score} pts  ({sum(n_correct)}/63 correct)')
            else:
                year_bracket_scores[year] = None
                print('(current year, no score)')

            # Re-generate HTML with correct ok/ng markings
            html_str = format_bracket_html(
                data_root=REPO_ROOT,
                year=year,
                pred_teams_by_round=pred_teams,
                pred_seeds_by_round=pred_seeds,
                pred_probs_by_round=pred_probs,
                correct_by_round=correct,
                num_correct_by_round=n_correct,
                total_score=score,
                is_current=is_current,
                model_key='ensemble',
                feat_bases=model_info.get('feature_bases', []),
                ff_pairings=ff_pairings,
                votes_by_round=winner_votes,
                n_components=len(components),
            )
            (ens_dir / f'{year}.html').write_text(html_str, encoding='utf-8')

        avg_score = total_score / num_eval_years if num_eval_years else 0.0
        print(f'\n  Bracket stats: {num_eval_years} years evaluated, '
              f'avg score={avg_score:.1f}')

        rebuild_summary(ens_dir, model_info, year_bracket_scores,
                        total_correct_by_round, total_score, num_eval_years)
        print(f'  summary.txt updated for {ens_dir.name}')

    print('\nDone.')


if __name__ == '__main__':
    main()
