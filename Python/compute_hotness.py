"""
compute_hotness.py

Populates Data/HotnessBartTorvikData/ with per-team *hotness* — the numeric
difference between the 2-week rolling stats and the full-season stats:

    hotness = 2WeekBartTorivkData[col] - BartTorvikData[col]

A positive value means a team is performing *better* recently than their
season average (they're "hot"); negative means they've cooled off.

Non-numeric / identity columns (Team, Seed, Conf, Rk, G, Rec, ConfRec,
Wins, Losses, ConfWins, ConfLosses) are carried over from the full-season
file unchanged.  WinPct and ConfWinPct are also differenced.

Usage:
    python compute_hotness.py [--years 2025 2026 ...]
    python compute_hotness.py          # processes all available years
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BT_DIR      = ROOT / 'Data' / 'BartTorvikData'
TWOWEEK_DIR = ROOT / 'Data' / '2WeekBartTorivkData'
OUT_DIR     = ROOT / 'Data' / 'HotnessBartTorvikData'

# Columns carried through as-is (identity / count columns).
ID_COLS = ['Rk', 'Team', 'Seed', 'Conf', 'G', 'Rec', 'ConfRec',
           'Wins', 'Losses', 'ConfWins', 'ConfLosses']

# Numeric columns to difference (2week − full-season).
DIFF_COLS = [
    'WinPct', 'ConfWinPct',
    'AdjO', 'Rk_AdjO',
    'AdjD', 'Rk_AdjD',
    'Barthag', 'Rk_Barthag',
    'EFG%', 'Rk_EFG%',
    'EFGD%', 'Rk_EFGD%',
    'TOR', 'Rk_TOR',
    'TORD', 'Rk_TORD',
    'ORB', 'Rk_ORB',
    'DRB', 'Rk_DRB',
    'FTR', 'Rk_FTR',
    'FTRD', 'Rk_FTRD',
    '2P%', 'Rk_2P%',
    '2P%D', 'Rk_2P%D',
    '3P%', 'Rk_3P%',
    '3P%D', 'Rk_3P%D',
    '3PR', 'Rk_3PR',
    '3PRD', 'Rk_3PRD',
    'AdjT', 'Rk_AdjT',
    'WAB', 'Rk_WAB',
]

OUT_COLS = ID_COLS + DIFF_COLS


def load_csv(path: Path) -> dict:
    """Load a BartTorvik CSV; return {team_name: row_dict}."""
    teams = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            teams[row['Team'].strip()] = row
    return teams


def safe_float(val: str):
    try:
        return float(str(val).strip().lstrip('+'))
    except (ValueError, AttributeError):
        return None


def process_year(year: int) -> int:
    """
    Compute hotness for one year.  Returns number of teams written.
    """
    bt_path = BT_DIR / f'{year}.csv'
    tw_path = TWOWEEK_DIR / f'{year}.csv'

    if not bt_path.exists():
        print(f'  [{year}] SKIP — {bt_path.name} not found in BartTorvikData')
        return 0
    if not tw_path.exists():
        print(f'  [{year}] SKIP — {tw_path.name} not found in 2WeekBartTorivkData')
        return 0

    bt_rows = load_csv(bt_path)
    tw_rows = load_csv(tw_path)

    shared = sorted(set(bt_rows) & set(tw_rows))
    missing_in_bt = set(tw_rows) - set(bt_rows)
    missing_in_tw = set(bt_rows) - set(tw_rows)

    if missing_in_bt:
        print(f'  [{year}] {len(missing_in_bt)} team(s) in 2-week only (skipped): '
              + ', '.join(sorted(missing_in_bt)))
    if missing_in_tw:
        print(f'  [{year}] {len(missing_in_tw)} team(s) in full-season only (skipped): '
              + ', '.join(sorted(missing_in_tw)))

    out_path = OUT_DIR / f'{year}.csv'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        writer.writeheader()

        for team in shared:
            bt  = bt_rows[team]
            tw  = tw_rows[team]
            row: dict = {}

            # Identity columns from full-season file.
            for col in ID_COLS:
                row[col] = bt.get(col, '')

            # Numeric diff columns.
            for col in DIFF_COLS:
                bt_val = safe_float(bt.get(col, ''))
                tw_val = safe_float(tw.get(col, ''))
                if bt_val is not None and tw_val is not None:
                    diff = round(tw_val - bt_val, 4)
                    row[col] = f'+{diff}' if diff > 0 else str(diff)
                else:
                    row[col] = ''

            writer.writerow(row)

    print(f'  [{year}] {len(shared)} teams written → {out_path.relative_to(ROOT)}')
    return len(shared)


def main():
    parser = argparse.ArgumentParser(description='Compute BartTorvik hotness data.')
    parser.add_argument('--years', nargs='+', type=int,
                        help='Years to process (default: all available).')
    args = parser.parse_args()

    if args.years:
        years = sorted(args.years)
    else:
        available = {int(p.stem) for p in BT_DIR.glob('*.csv')} & \
                    {int(p.stem) for p in TWOWEEK_DIR.glob('*.csv')}
        years = sorted(available)

    if not years:
        print('No matching year CSVs found in both directories.', file=sys.stderr)
        sys.exit(1)

    print(f'Processing {len(years)} year(s): {years}\n')
    total = 0
    for yr in years:
        total += process_year(yr)
    print(f'\nDone. {total} team-rows written across {len(years)} year(s).')


if __name__ == '__main__':
    main()
