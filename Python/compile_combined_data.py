"""
compile_combined_data.py

Compiles GameCombinedData and/or BracketCombinedData for all (or a single) year
by merging game/bracket data with KenPom stats using pd.merge (avoids the silent
row-drop bug from chained .join() calls in the older scripts).

Replaces:
    compile_combined_game_data.py
    compile_combined_bracket_data.py
    compile_combined_bracket_data2.py

Usage:
    python3 compile_combined_data.py                      # all years, both types
    python3 compile_combined_data.py --year 2025          # single year, both types
    python3 compile_combined_data.py --type bracket       # all years, bracket only
    python3 compile_combined_data.py --type game          # all years, game only
    python3 compile_combined_data.py -y 2025 -t bracket   # single year, bracket only
"""

import argparse
import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_YEARS = [y for y in range(2012, 2026) if y != 2020]
BRACKET_ROUNDS = range(1, 7)  # rounds 1–6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rename_matchup_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw Team1/Team2-style columns to the double-underscore convention."""
    return df.rename(columns={
        'Team1':       'Team__1',
        'Team1_Seed':  'Seed__1',
        'Team1_Score': 'Score__1',
        'Team2':       'Team__2',
        'Team2_Seed':  'Seed__2',
        'Team2_Score': 'Score__2',
        'WinningTeam': 'Winning_Team',
        'Team1_Win':   'Win__1',
    })


def attach_kenpom(df: pd.DataFrame, df_kp: pd.DataFrame) -> pd.DataFrame:
    """
    Merge KenPom stats into a matchup DataFrame that already uses the
    double-underscore column convention (Team__1, Team__2, …).

    KenPom's Seed column is dropped before merging to keep the seed values
    that are already present in the matchup data and to avoid duplicate columns.
    """
    kp = df_kp.drop(columns=['Seed'], errors='ignore')

    kp1 = kp.add_suffix('__1')   # Team → Team__1, AdjEM → AdjEM__1, …
    kp2 = kp.add_suffix('__2')

    result = df.merge(kp1, on='Team__1', how='inner')
    result = result.merge(kp2, on='Team__2', how='inner')
    return result.reset_index(drop=True)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place __1 KenPom columns first, then __2, then remaining metadata columns."""
    cols = df.columns.tolist()
    cols1 = [c for c in cols if '__1' in c and c != 'Win__1']
    cols2 = [c for c in cols if '__2' in c]
    rest  = [c for c in cols if c not in cols1 and c not in cols2]
    return df[cols1 + cols2 + rest]


# ---------------------------------------------------------------------------
# Per-year compile functions
# ---------------------------------------------------------------------------

def compile_game_year(data_root: Path, year: int) -> pd.DataFrame:
    """Return the combined game DataFrame for a single year."""
    game_path = data_root / 'Data' / 'GameData' / f'{year}.csv'
    kp_path   = data_root / 'Data' / 'KenPomData' / f'{year}.csv'

    df_games = pd.read_csv(game_path)
    df_kp    = pd.read_csv(kp_path)

    df_games = rename_matchup_cols(df_games)
    df_combined = attach_kenpom(df_games, df_kp)
    df_combined = reorder_columns(df_combined)
    return df_combined


def compile_bracket_year(data_root: Path, year: int) -> None:
    """Write BracketCombinedData CSVs for all available rounds of a single year."""
    kp_path = data_root / 'Data' / 'KenPomData' / f'{year}.csv'
    df_kp   = pd.read_csv(kp_path)

    out_dir = data_root / 'Data' / 'BracketCombinedData' / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rnd in BRACKET_ROUNDS:
        bracket_path = (
            data_root / 'Data' / 'BracketData' / str(year) / f'Round{rnd}_{year}.csv'
        )
        if not bracket_path.exists():
            print(f'    Round {rnd}: source file not found, skipping')
            continue

        df_round = pd.read_csv(bracket_path)
        df_round = rename_matchup_cols(df_round)
        df_combined = attach_kenpom(df_round, df_kp)
        df_combined = reorder_columns(df_combined)

        out_path = out_dir / f'Round{rnd}_{year}.csv'
        df_combined.to_csv(out_path, index=False)
        print(f'    Round {rnd}: {df_combined.shape[0]} rows → {out_path.relative_to(data_root)}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Compile GameCombinedData and/or BracketCombinedData by joining '
            'game/bracket data with KenPom stats.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--year', '-y',
        type=int,
        default=None,
        help='Process only this year (default: all years).',
    )
    parser.add_argument(
        '--type', '-t',
        choices=['game', 'bracket', 'both'],
        default='both',
        help='Which combined data to compile (default: both).',
    )
    args = parser.parse_args()

    data_root  = Path(__file__).resolve().parents[1]
    years      = [args.year] if args.year else ALL_YEARS
    do_game    = args.type in ('game', 'both')
    do_bracket = args.type in ('bracket', 'both')

    # -----------------------------------------------------------------------
    # GameCombinedData
    # -----------------------------------------------------------------------
    if do_game:
        print('=== Compiling GameCombinedData ===')
        df_all = pd.DataFrame()
        for year in years:
            game_path = data_root / 'Data' / 'GameData' / f'{year}.csv'
            kp_path   = data_root / 'Data' / 'KenPomData' / f'{year}.csv'
            if not game_path.exists():
                print(f'  {year}: GameData/{year}.csv not found, skipping')
                continue
            if not kp_path.exists():
                print(f'  {year}: KenPomData/{year}.csv not found, skipping')
                continue

            print(f'  {year}:', end=' ', flush=True)
            df = compile_game_year(data_root, year)
            print(f'{df.shape[0]} rows')

            out_path = data_root / 'Data' / 'GameCombinedData' / f'{year}.csv'
            df.to_csv(out_path, index=False)
            df_all = pd.concat([df_all, df], ignore_index=True)

        # Regenerate All.csv only when processing a full run (no --year filter).
        if not args.year:
            all_path = data_root / 'Data' / 'GameCombinedData' / 'All.csv'
            df_all.to_csv(all_path, index=False)
            print(f'  All.csv: {df_all.shape[0]} total rows')

    # -----------------------------------------------------------------------
    # BracketCombinedData
    # -----------------------------------------------------------------------
    if do_bracket:
        print('\n=== Compiling BracketCombinedData ===')
        for year in years:
            kp_path = data_root / 'Data' / 'KenPomData' / f'{year}.csv'
            bracket_dir = data_root / 'Data' / 'BracketData' / str(year)
            if not kp_path.exists():
                print(f'  {year}: KenPomData/{year}.csv not found, skipping')
                continue
            if not bracket_dir.exists():
                print(f'  {year}: BracketData/{year}/ not found, skipping')
                continue

            print(f'  {year}:')
            compile_bracket_year(data_root, year)


if __name__ == '__main__':
    main()
