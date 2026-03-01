"""
determine_winner.py

Iterates over every BracketData CSV and recomputes WinningTeam and Team1_Win
from the actual scores (Team1_Score vs Team2_Score), overwriting the files
in place.

Usage:
    python3 determine_winner.py               # all years
    python3 determine_winner.py --year 2025   # single year
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / 'Data'
ALL_YEARS = [y for y in range(2012, 2026) if y != 2020]


def fix_winners(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    team1_won = df['Team1_Score'] > df['Team2_Score']
    df['WinningTeam'] = df['Team1'].where(team1_won, df['Team2'])
    df['Team1_Win'] = team1_won
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Recompute WinningTeam and Team1_Win from scores in BracketData CSVs.',
    )
    parser.add_argument('--year', '-y', type=int, default=None,
                        help='Process only this year (default: all years).')
    args = parser.parse_args()
    years = [args.year] if args.year else ALL_YEARS

    for year in years:
        bracket_dir = DATA_ROOT / 'BracketData' / str(year)
        if not bracket_dir.exists():
            print(f'{year}: directory not found, skipping')
            continue

        total_fixed = 0
        for path in sorted(bracket_dir.glob('*.csv')):
            df = pd.read_csv(path)

            # Skip rounds where scores are not available (all zeros / NaN)
            if df['Team1_Score'].isna().all() or (df['Team1_Score'] == 0).all():
                print(f'  {path.name}: no scores, skipping')
                continue

            df_fixed = fix_winners(df)
            changed = (df_fixed['Team1_Win'] != df['Team1_Win']).sum()
            df_fixed.to_csv(path, index=False)
            total_fixed += changed

        print(f'{year}: {total_fixed} row(s) corrected across all rounds')


if __name__ == '__main__':
    main()
