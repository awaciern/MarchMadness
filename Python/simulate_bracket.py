"""
simulate_bracket.py

Monte Carlo simulation of an NCAA tournament bracket using a saved model pickle.

For each iteration, games are decided probabilistically: the model's predicted
win probability for Team__1 is used as the coin-flip weight rather than always
picking the favourite.  Running many iterations gives each team an empirical
probability of advancing to every round.

Output
------
- Terminal: formatted table sorted by championship likelihood
- <output_dir>/<year>_<N>iters.html  — styled HTML table
- <output_dir>/<year>_<N>iters.csv   — flat CSV

Default output directory: Simulations/<model_folder_name>/

Usage
-----
    python3 Python/simulate_bracket.py \\
        --model Predictions/random_forest_767_KP_.../model.pkl \\
        --year 2024 --num-iters 10000

    python3 Python/simulate_bracket.py \\
        --model Predictions/.../model.pkl --year 2026 --num-iters 5000 \\
        --final-four-pairings "0-2,1-3" --seed 42
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import helpers from predict_brackets.py (same package directory)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_brackets import (
    load_bracket_round,
    load_kenpom,
    load_barttorvik,
    attach_kenpom,
    attach_barttorvik,
    apply_label_encoders,
    derive_ff_pairings_from_data,
    parse_ff_pairings_arg,
    ALL_YEARS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_LABELS  = ['R32', 'R16', 'E8', 'FF', 'Final', 'Champ']
ROUND_FULL    = ['Round of 32', 'Sweet 16', 'Elite Eight',
                 'Final Four', 'Championship', 'Champion']
# Actual-result display labels (extended with first-round exit)
ACTUAL_LABELS = ['R1'] + ROUND_LABELS  # index 0 = first-round exit
_ACTUAL_ORDER = {lbl: i for i, lbl in enumerate(ACTUAL_LABELS)}

# Colors for actual-result cells in HTML
_ACTUAL_CSS = {
    'Champ': 'act-champ',
    'Final': 'act-final',
    'FF':    'act-ff',
    'E8':    'act-e8',
    'R16':   'act-r16',
    'R32':   'act-r32',
    'R1':    'act-r1',
}


# ---------------------------------------------------------------------------
# Actual results
# ---------------------------------------------------------------------------

def get_actual_results(data_root: Path, year: int) -> dict:
    """
    Return {team_name: label} where label is the furthest round the team
    actually advanced to in `year`.  Teams that lost in Round 1 are mapped
    to 'R1'.  Returns an empty dict if no bracket data is available.

    Labels follow ROUND_LABELS: 'R32', 'R16', 'E8', 'FF', 'Final', 'Champ'.
    (R32 = won first-round game; R16 = made Sweet 16; etc.)
    """
    _rnd_to_label = {i + 1: lbl for i, lbl in enumerate(ROUND_LABELS)}  # 1→R32 … 6→Champ
    actual: dict = {}

    # Seed all participants from Round 1
    try:
        r1_df = load_bracket_round(data_root, year, 1)
    except Exception:
        return {}

    for _, row in r1_df.iterrows():
        actual.setdefault(row['Team__1'], 'R1')
        actual.setdefault(row['Team__2'], 'R1')

    # Walk rounds 1-6, updating each winner's furthest label
    for rnd in range(1, 7):
        try:
            df = load_bracket_round(data_root, year, rnd)
        except Exception:
            break
        if 'Winning_Team' not in df.columns:
            break
        label = _rnd_to_label.get(rnd)
        if label is None:
            break
        for team in df['Winning_Team'].dropna():
            if team in actual:
                actual[team] = label

    return actual


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _simulate_one(
    model,
    r1_teams1: list,
    r1_teams2: list,
    r1_proba: np.ndarray,
    df_kp: pd.DataFrame,
    df_bt,                  # pd.DataFrame or None
    feature_list: list,
    cat_encoders: dict,
    team_seed_map: dict,
    ff_pairings: list,
    needs_bt: bool,
    draws_r1: np.ndarray,   # pre-drawn randoms for round 1
    rng: np.random.Generator,
    year_sc=None,           # fitted StandardScaler for this year (or None)
    norm_cols: list = None, # numeric columns to normalise
) -> dict:
    """
    Run one complete bracket simulation.
    Returns {round_number: [winning_team_names]} for rounds 1–6.
    """
    results = {}
    prev_winners: list = []

    for rnd in range(1, 7):
        if rnd == 1:
            teams1 = r1_teams1
            teams2 = r1_teams2
            proba  = r1_proba
            draws  = draws_r1
        else:
            # Build matchups from previous round's winners
            if rnd == 5:
                matchup_teams = [
                    (prev_winners[ff_pairings[0][0]], prev_winners[ff_pairings[0][1]]),
                    (prev_winners[ff_pairings[1][0]], prev_winners[ff_pairings[1][1]]),
                ]
            else:
                matchup_teams = [
                    (prev_winners[i], prev_winners[i + 1])
                    for i in range(0, len(prev_winners), 2)
                ]

            df_match = pd.DataFrame(matchup_teams, columns=['Team__1', 'Team__2'])
            df_rnd = attach_kenpom(df_match, df_kp)
            if needs_bt and df_bt is not None:
                df_rnd = attach_barttorvik(df_rnd, df_bt)
            df_rnd['Seed__1'] = df_rnd['Team__1'].map(team_seed_map)
            df_rnd['Seed__2'] = df_rnd['Team__2'].map(team_seed_map)

            df_enc = apply_label_encoders(df_rnd, cat_encoders) if cat_encoders else df_rnd
            if year_sc is not None and norm_cols:
                _avail = [c for c in norm_cols if c in df_enc.columns]
                if _avail:
                    df_enc = df_enc.copy()
                    df_enc[_avail] = year_sc.transform(df_enc[_avail])
            proba  = model.predict_proba(df_enc[feature_list])
            teams1 = df_rnd['Team__1'].tolist()
            teams2 = df_rnd['Team__2'].tolist()
            draws  = rng.random(len(teams1))

        winners = [
            teams1[k] if draws[k] < proba[k, 1] else teams2[k]
            for k in range(len(teams1))
        ]
        results[rnd] = winners
        prev_winners = winners

    return results


def run_simulations(
    model,
    data_root: Path,
    year: int,
    ff_pairings: list,
    feature_list: list,
    cat_encoders: dict,
    num_iters: int,
    seed: int = None,
    norm_info: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run `num_iters` Monte Carlo bracket simulations for `year`.

    Returns a DataFrame (64 rows) with columns:
        Team, Seed, R16, S16, E8, FF, Final, Champ
    Values are probabilities (0–1), sorted by Champ desc → FF desc → … → R16 desc.
    """
    needs_bt = any(f.startswith('BT__') for f in feature_list)

    # --- Load fixed data --------------------------------------------------
    r1_df_raw = load_bracket_round(data_root, year, 1)

    # Build team → seed map from Round 1
    team_seed_map: dict = {}
    for _, row in r1_df_raw.iterrows():
        team_seed_map[row['Team__1']] = row['Seed__1']
        team_seed_map[row['Team__2']] = row['Seed__2']

    # All 64 teams in bracket order (Team__1, Team__2 per game)
    all_teams: list = []
    for _, row in r1_df_raw.iterrows():
        for t in (row['Team__1'], row['Team__2']):
            if t not in all_teams:
                all_teams.append(t)

    # Pre-load stat files once
    df_kp = load_kenpom(data_root, year)
    df_bt = load_barttorvik(data_root, year) if needs_bt else None

    # Pre-compute Round 1 feature matrix & probabilities (matchups are fixed)
    df_r1_kp = attach_kenpom(r1_df_raw[['Team__1', 'Team__2']].copy(), df_kp)
    if needs_bt and df_bt is not None:
        df_r1_kp = attach_barttorvik(df_r1_kp, df_bt)
    df_r1_kp['Seed__1'] = df_r1_kp['Team__1'].map(team_seed_map)
    df_r1_kp['Seed__2'] = df_r1_kp['Team__2'].map(team_seed_map)
    # Derive per-year scaler (if model was trained with --norm-years)
    year_sc   = None
    norm_cols: list = []
    if norm_info:
        year_sc   = norm_info['by_year'].get(year, norm_info.get('fallback'))
        norm_cols = norm_info.get('cols', [])

    df_r1_enc = apply_label_encoders(df_r1_kp, cat_encoders) if cat_encoders else df_r1_kp
    if year_sc is not None and norm_cols:
        _avail = [c for c in norm_cols if c in df_r1_enc.columns]
        if _avail:
            df_r1_enc = df_r1_enc.copy()
            df_r1_enc[_avail] = year_sc.transform(df_r1_enc[_avail])
    r1_proba  = model.predict_proba(df_r1_enc[feature_list])

    # Aligned team lists for Round 1 (same row order as r1_proba)
    r1_teams1 = df_r1_kp['Team__1'].tolist()
    r1_teams2 = df_r1_kp['Team__2'].tolist()

    # Verify model has predict_proba
    if not hasattr(model, 'predict_proba'):
        raise RuntimeError(
            'The saved model does not support predict_proba. '
            'Re-run predict_brackets.py to generate a probability-capable model '
            '(e.g. for SVC, pass probability=True).'
        )

    # --- Run simulations --------------------------------------------------
    rng = np.random.default_rng(seed)
    round_wins: dict = defaultdict(lambda: defaultdict(int))  # team → rnd → count

    _prog_interval = max(1, num_iters // 100)
    print(f'Running {num_iters:,} simulations for {year}...', flush=True)
    for i in range(num_iters):
        if (i + 1) % _prog_interval == 0 or i + 1 == num_iters:
            # PROGRESS: prefix is parsed by the web UI for the progress bar;
            # \r keeps the terminal display clean (overwrites same line).
            print(f'PROGRESS:{i+1}/{num_iters}', end='\r', flush=True)

        # Pre-draw Round 1 randoms (same rng stream for reproducibility)
        draws_r1 = rng.random(len(r1_teams1))
        sim = _simulate_one(
            model, r1_teams1, r1_teams2, r1_proba,
            df_kp, df_bt, feature_list, cat_encoders,
            team_seed_map, ff_pairings, needs_bt, draws_r1, rng,
            year_sc=year_sc, norm_cols=norm_cols,
        )
        for rnd, winners in sim.items():
            for team in winners:
                round_wins[team][rnd] += 1

    print(f'  {num_iters}/{num_iters} — done.    ', flush=True)

    # --- Build results DataFrame ------------------------------------------
    rows = []
    for team in all_teams:
        row = {'Team': team, 'Seed': team_seed_map.get(team, '?')}
        for rnd, lbl in enumerate(ROUND_LABELS, start=1):
            row[lbl] = round_wins[team][rnd] / num_iters
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort: Champ desc, then FF, E8, S16, R16 desc
    df = df.sort_values(list(reversed(ROUND_LABELS)), ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    if v == 0:
        return '—'
    if v >= 0.9995:
        return '100%'
    if v < 0.0005:
        return '<0.1%'
    return f'{v:.1%}'


def to_html(
    df: pd.DataFrame,
    year: int,
    num_iters: int,
    model_key: str,
    actual: Optional[dict] = None,
) -> str:
    has_actual = bool(actual)
    extra_th   = '<th class="act-hdr">Actual</th>' if has_actual else ''
    header_cells = ''.join(
        f'<th>{h}</th>' for h in ['#', 'Seed', 'Team']
    ) + extra_th + ''.join(
        f'<th>{h}</th>' for h in ROUND_FULL
    )
    rows_html = ''
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            seed = int(float(row['Seed']))
        except (ValueError, TypeError):
            seed = '?'
        cells = (
            f'<td class="rank">{rank}</td>'
            f'<td class="seed">[{seed}]</td>'
            f'<td class="team">{row["Team"]}</td>'
        )
        if has_actual:
            act_lbl = actual.get(row['Team'], '')
            act_cls = _ACTUAL_CSS.get(act_lbl, 'act-r1') if act_lbl else ''
            cells += f'<td class="actual {act_cls}">{act_lbl}</td>'
        for lbl in ROUND_LABELS:
            v = row[lbl]
            if v >= 0.50:
                cls = 'p-high'
            elif v >= 0.15:
                cls = 'p-med'
            elif v > 0:
                cls = 'p-low'
            else:
                cls = 'p-zero'
            cells += f'<td class="{cls}">{_fmt_pct(v)}</td>'
        rows_html += f'<tr>{cells}</tr>\n'

    return f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{year} Monte Carlo Simulation \u2013 {model_key}</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; font-size: 12px;
       background: #0f172a; color: #e2e8f0; padding: 24px; }}
h1 {{ font-size: 18px; color: #fbbf24; margin-bottom: 4px; }}
.meta {{ color: #64748b; font-size: 11px; margin-bottom: 16px; }}
table {{ border-collapse: collapse; }}
th {{ background: #1e293b; color: #64748b; font-size: 10px; text-transform: uppercase;
      letter-spacing: .6px; padding: 8px 12px; border-bottom: 2px solid #334155;
      text-align: right; white-space: nowrap; }}
th:nth-child(2), th:nth-child(3), th:nth-child(4) {{ text-align: left; }}
.act-hdr {{ color: #a78bfa !important; }}
td {{ padding: 5px 12px; border-bottom: 1px solid #1e293b; text-align: right;
      font-variant-numeric: tabular-nums; white-space: nowrap; }}
tr:hover td {{ background: #1e293b; }}
.rank   {{ color: #475569; font-size: 10px; text-align: right; }}
.seed   {{ color: #64748b; font-size: 10px; text-align: left; }}
.team   {{ color: #e2e8f0; font-weight: 500; text-align: left; min-width: 160px; }}
.actual {{ font-weight: 600; text-align: left; min-width: 52px; }}
.act-champ {{ color: #fbbf24; }}
.act-final {{ color: #f97316; }}
.act-ff    {{ color: #a78bfa; }}
.act-e8    {{ color: #38bdf8; }}
.act-s16   {{ color: #86efac; }}
.act-r32   {{ color: #94a3b8; }}
.act-r1    {{ color: #475569; }}
.p-high {{ color: #86efac; font-weight: 700; }}
.p-med  {{ color: #fde68a; }}
.p-low  {{ color: #94a3b8; }}
.p-zero {{ color: #334155; }}
</style></head>
<body>
<h1>{year} NCAA Tournament \u2014 Monte Carlo Simulation</h1>
<div class="meta">
  Model: <strong style="color:#e2e8f0">{model_key}</strong>
  &nbsp;|&nbsp; Iterations: <strong style="color:#e2e8f0">{num_iters:,}</strong>
  &nbsp;|&nbsp; Values show probability of advancing to each round
</div>
<table>
  <thead><tr>{header_cells}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</body></html>'''


def to_terminal(df: pd.DataFrame, actual: Optional[dict] = None):
    team_w  = max(df['Team'].str.len().max(), 4)
    act_col = bool(actual)
    act_w   = 6  # width for 'Actual' column
    hdr = f"{'#':>3}  {'Seed':>4}  {'Team':<{team_w}}"
    if act_col:
        hdr += f"  {'Actual':<{act_w}}"
    hdr += ''.join(f'  {lbl:>7}' for lbl in ROUND_LABELS)
    print(hdr)
    print('-' * len(hdr))
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            seed = int(float(row['Seed']))
        except (ValueError, TypeError):
            seed = '?'
        line = f'{rank:>3}  {seed:>4}  {row["Team"]:<{team_w}}'
        if act_col:
            act_lbl = actual.get(row['Team'], '')
            line += f'  {act_lbl:<{act_w}}'
        line += ''.join(f'  {_fmt_pct(row[lbl]):>7}' for lbl in ROUND_LABELS)
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo bracket simulation using a saved model pickle.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--model', '-m', required=True,
        help='Path to the model .pkl file (inside a Predictions/<dir>/ folder).',
    )
    parser.add_argument(
        '--year', '-y', type=int, required=True,
        help='Tournament year to simulate.',
    )
    parser.add_argument(
        '--num-iters', '-n', type=int, default=1000,
        help='Number of Monte Carlo iterations (default: 1000).',
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Output directory. Default: Simulations/<model_folder>/ under repo root.',
    )
    parser.add_argument(
        '--final-four-pairings', default=None,
        help=(
            'Override FF bracket pairings as "i-j,k-l" (e.g. "0-2,1-3"). '
            'Past years auto-derive from data; unknown years default to "0-1,2-3".'
        ),
    )
    parser.add_argument(
        '--data-root', '-d', default=None,
        help='Repo root containing Data/. Inferred from pickle path if omitted.',
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Integer RNG seed for reproducibility.',
    )
    args = parser.parse_args()

    # --- Load pickle ------------------------------------------------------
    pkl_path = Path(args.model).resolve()
    if not pkl_path.exists():
        print(f'ERROR: pickle not found: {pkl_path}', file=sys.stderr)
        sys.exit(1)

    with open(pkl_path, 'rb') as fh:
        payload = pickle.load(fh)

    model        = payload['model']
    model_key    = payload['model_key']
    feature_list = payload['feature_list']
    cat_encoders = payload.get('cat_encoders', {})
    norm_info    = payload.get('norm_info')

    # --- Infer data root --------------------------------------------------
    # Expected layout: <repo_root>/Predictions/<model_dir>/model.pkl
    if args.data_root:
        data_root = Path(args.data_root).resolve()
    else:
        data_root = pkl_path.parent.parent.parent
        if not (data_root / 'Data').is_dir():
            print(
                'ERROR: could not infer repo root from pickle path. '
                'Use --data-root.',
                file=sys.stderr,
            )
            sys.exit(1)

    # --- FF pairings ------------------------------------------------------
    if args.final_four_pairings:
        ff_pairings = parse_ff_pairings_arg(args.final_four_pairings)
    elif args.year in ALL_YEARS:
        try:
            ff_pairings = derive_ff_pairings_from_data(data_root, args.year)
        except Exception as e:
            print(f'WARNING: could not derive FF pairings ({e}), using 0-1,2-3')
            ff_pairings = [(0, 1), (2, 3)]
    else:
        ff_pairings = [(0, 1), (2, 3)]

    print(f'Model:        {model_key}')
    print(f'Year:         {args.year}')
    print(f'Iterations:   {args.num_iters:,}')
    print(f'FF pairings:  {ff_pairings}')
    if args.seed is not None:
        print(f'RNG seed:     {args.seed}')
    print()

    # --- Run simulations --------------------------------------------------
    df_results = run_simulations(
        model=model,
        data_root=data_root,
        year=args.year,
        ff_pairings=ff_pairings,
        feature_list=feature_list,
        cat_encoders=cat_encoders,
        num_iters=args.num_iters,
        seed=args.seed,
        norm_info=norm_info,
    )

    # --- Actual results (only for historical years with bracket data) ------
    actual: dict = {}
    if args.year in ALL_YEARS:
        actual = get_actual_results(data_root, args.year)
        if actual:
            print(f'Loaded actual results for {args.year}.')
        else:
            print(f'No actual bracket data found for {args.year}.')
    print()

    # --- Print to terminal ------------------------------------------------
    to_terminal(df_results, actual or None)

    # --- Write output files -----------------------------------------------
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = data_root / 'Simulations' / pkl_path.parent.name

    out_dir.mkdir(parents=True, exist_ok=True)

    stem     = f'{args.year}_{args.num_iters}iters'
    html_out = out_dir / f'{stem}.html'
    csv_out  = out_dir / f'{stem}.csv'

    html_out.write_text(
        to_html(df_results, args.year, args.num_iters, model_key, actual or None),
        encoding='utf-8',
    )
    df_results.to_csv(csv_out, index=False)

    print(f'\nHTML saved to: {html_out}')
    print(f'CSV  saved to: {csv_out}')


if __name__ == '__main__':
    main()
