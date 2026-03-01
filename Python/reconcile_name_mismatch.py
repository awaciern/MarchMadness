"""
reconcile_name_mismatch.py

Normalises team names in KenPomData, BracketData, and GameData CSVs so every
source uses the same canonical name for each team.

Canonical names (ground truth) are taken from GameData.  All known aliases are
maintained in:

    Data/TeamNames/team_names.csv

Each row in that CSV has the form:
    Canonical, Alias1, Alias2, ...

To add a new alias, simply add it to the appropriate row in team_names.csv.

Run this script BEFORE compile_combined_data.py.

Usage:
    python3 reconcile_name_mismatch.py
    python3 reconcile_name_mismatch.py --year 2025   # single year only
"""

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parents[1] / 'Data'
ALL_YEARS = [y for y in range(2012, 2026) if y != 2020]

TEAM_NAMES_CSV = DATA_ROOT / 'TeamNames' / 'team_names.csv'


def load_alias_map() -> dict:
    """
    Read team_names.csv and return a dict mapping every alias → canonical name.
    The first column is the canonical; all remaining non-empty columns are aliases.
    """
    df = pd.read_csv(TEAM_NAMES_CSV, dtype=str)
    alias_map: dict = {}
    canonical_col = df.columns[0]
    for _, row in df.iterrows():
        canonical = row[canonical_col].strip()
        for col in df.columns[1:]:
            alias = str(row[col]).strip()
            if alias and alias != 'nan':
                alias_map[alias] = canonical
    return alias_map


def apply_rename(df: pd.DataFrame, name_map: dict, cols: list) -> pd.DataFrame:
    """Replace values in the given columns using name_map."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(name_map)
    return df


# ---------------------------------------------------------------------------
# Per-source reconciliation
# ---------------------------------------------------------------------------

KP_COLS      = ['Team']
BRACKET_COLS = ['Team1', 'Team2', 'WinningTeam']
GAME_COLS    = ['Team1', 'Team2', 'WinningTeam']


def reconcile_kenpom(years, alias_map):
    print('=== KenPom CSVs ===')
    for year in years:
        path = DATA_ROOT / 'KenPomData' / f'{year}.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df_new = apply_rename(df, alias_map, KP_COLS)
        changed = (df_new['Team'] != df['Team']).sum()
        df_new.to_csv(path, index=False)
        print(f'  {year}: {changed} name(s) updated')


def reconcile_bracket(years, alias_map):
    print('\n=== BracketData CSVs ===')
    for year in years:
        bracket_dir = DATA_ROOT / 'BracketData' / str(year)
        if not bracket_dir.exists():
            continue
        total_changed = 0
        for path in sorted(bracket_dir.glob('*.csv')):
            df = pd.read_csv(path)
            df_new = apply_rename(df, alias_map, BRACKET_COLS)
            changed = sum(
                (df_new[c] != df[c]).sum() for c in BRACKET_COLS if c in df.columns
            )
            if changed:
                df_new.to_csv(path, index=False)
                total_changed += changed
        print(f'  {year}: {total_changed} name(s) updated')


def reconcile_games(years, alias_map):
    print('\n=== GameData CSVs ===')
    game_dir = DATA_ROOT / 'GameData'
    for year in years:
        path = game_dir / f'{year}.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df_new = apply_rename(df, alias_map, GAME_COLS)
        changed = sum(
            (df_new[c] != df[c]).sum() for c in GAME_COLS if c in df.columns
        )
        if changed:
            df_new.to_csv(path, index=False)
        print(f'  {year}: {changed} name(s) updated')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Normalise team names across KenPom, BracketData, and GameData CSVs.',
    )
    parser.add_argument(
        '--year', '-y',
        type=int,
        default=None,
        help='Process only this year (default: all years).',
    )
    args = parser.parse_args()
    years = [args.year] if args.year else ALL_YEARS

    alias_map = load_alias_map()
    print(f'Loaded {len(alias_map)} alias → canonical mappings from {TEAM_NAMES_CSV.relative_to(DATA_ROOT.parent)}\n')

    reconcile_kenpom(years, alias_map)
    reconcile_bracket(years, alias_map)
    reconcile_games(years, alias_map)

    print('\nDone. Run compile_combined_data.py to rebuild combined CSVs.')


if __name__ == '__main__':
    main()
