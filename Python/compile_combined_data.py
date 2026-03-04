"""
compile_combined_data.py

Compiles GameCombinedData and/or BracketCombinedData for all (or a single) year
by merging game/bracket data with KenPom stats (KP__ prefix) and BartTorvik
stats (BT__ prefix) using pd.merge.  The __1 / __2 suffix further identifies
which team each column belongs to, e.g. KP__AdjEM__1 and BT__AdjOE__2.

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


def prefix_stats_cols(df: pd.DataFrame, prefix: str, drop_cols: list = None) -> pd.DataFrame:
    """Prefix every non-Team column with *prefix* (e.g. 'KP__' or 'BT__').
    Columns listed in drop_cols are removed before prefixing."""
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    rename_map = {c: f'{prefix}{c}' for c in df.columns if c != 'Team'}
    return df.rename(columns=rename_map)


def attach_kenpom(df: pd.DataFrame, df_kp: pd.DataFrame) -> pd.DataFrame:
    """
    Merge KenPom stats (KP__-prefixed) into a matchup DataFrame.
    Seed is dropped from KenPom – the matchup data already carries seeds.
    """
    kp = prefix_stats_cols(df_kp, 'KP__', drop_cols=['Seed'])

    kp1 = kp.add_suffix('__1')   # Team → Team__1, KP__AdjEM → KP__AdjEM__1
    kp2 = kp.add_suffix('__2')

    result = df.merge(kp1, on='Team__1', how='inner')
    result = result.merge(kp2, on='Team__2', how='inner')
    return result.reset_index(drop=True)


def attach_barttorvik(df: pd.DataFrame, df_bt: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BartTorvik stats (BT__-prefixed) into a matchup DataFrame.
    Uses a left join so rows are preserved even if a team is absent from
    BartTorvik data (e.g. play-in teams, pre-tournament year).
    """
    bt = prefix_stats_cols(df_bt, 'BT__', drop_cols=['Seed'])

    bt1 = bt.add_suffix('__1')   # Team → Team__1, BT__AdjOE → BT__AdjOE__1
    bt2 = bt.add_suffix('__2')

    result = df.merge(bt1, on='Team__1', how='left')
    result = result.merge(bt2, on='Team__2', how='left')
    return result.reset_index(drop=True)


def attach_barttorvik_2week(df: pd.DataFrame, df_bt2w: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 2-week BartTorvik stats (BT2W__-prefixed) into a matchup DataFrame.
    The 2-week snapshot has the same column schema as BartTorvikData.
    Uses a left join to preserve rows for teams absent from this source.
    """
    bt2w = prefix_stats_cols(df_bt2w, 'BT2W__', drop_cols=['Seed'])

    bt2w1 = bt2w.add_suffix('__1')
    bt2w2 = bt2w.add_suffix('__2')

    result = df.merge(bt2w1, on='Team__1', how='left')
    result = result.merge(bt2w2, on='Team__2', how='left')
    return result.reset_index(drop=True)


def attach_barttorvik_hotness(df: pd.DataFrame, df_hot: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Hotness BartTorvik stats (BTHOT__-prefixed) into a matchup DataFrame.
    Hotness values are the difference between 2-week and season-long BartTorvik
    stats (positive = improving, negative = declining).
    Uses a left join to preserve rows for teams absent from this source.
    """
    hot = prefix_stats_cols(df_hot, 'BTHOT__', drop_cols=['Seed'])

    hot1 = hot.add_suffix('__1')
    hot2 = hot.add_suffix('__2')

    result = df.merge(hot1, on='Team__1', how='left')
    result = result.merge(hot2, on='Team__2', how='left')
    return result.reset_index(drop=True)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place metadata first, then KP, BT, BT2W, BTHOT columns (each __1 before __2)."""
    cols = df.columns.tolist()
    kp1    = [c for c in cols if c.startswith('KP__')    and c.endswith('__1')]
    kp2    = [c for c in cols if c.startswith('KP__')    and c.endswith('__2')]
    bt1    = [c for c in cols if c.startswith('BT__')    and c.endswith('__1')]
    bt2    = [c for c in cols if c.startswith('BT__')    and c.endswith('__2')]
    bt2w1  = [c for c in cols if c.startswith('BT2W__')  and c.endswith('__1')]
    bt2w2  = [c for c in cols if c.startswith('BT2W__')  and c.endswith('__2')]
    bthot1 = [c for c in cols if c.startswith('BTHOT__') and c.endswith('__1')]
    bthot2 = [c for c in cols if c.startswith('BTHOT__') and c.endswith('__2')]
    ordered = kp1 + kp2 + bt1 + bt2 + bt2w1 + bt2w2 + bthot1 + bthot2
    rest = [c for c in cols if c not in ordered]
    return df[rest + ordered]


# ---------------------------------------------------------------------------
# Per-year compile functions
# ---------------------------------------------------------------------------

def compile_game_year(data_root: Path, year: int) -> pd.DataFrame:
    """Return the combined game DataFrame for a single year."""
    game_path  = data_root / 'Data' / 'GameData'             / f'{year}.csv'
    kp_path    = data_root / 'Data' / 'KenPomData'           / f'{year}.csv'
    bt_path    = data_root / 'Data' / 'BartTorvikData'       / f'{year}.csv'
    bt2w_path  = data_root / 'Data' / '2WeekBartTorivkData'  / f'{year}.csv'
    bthot_path = data_root / 'Data' / 'HotnessBartTorvikData' / f'{year}.csv'

    df_games = pd.read_csv(game_path)
    df_kp    = pd.read_csv(kp_path)

    df_games    = rename_matchup_cols(df_games)
    df_combined = attach_kenpom(df_games, df_kp)
    if bt_path.exists():
        df_bt       = pd.read_csv(bt_path)
        df_combined = attach_barttorvik(df_combined, df_bt)
    if bt2w_path.exists():
        df_bt2w     = pd.read_csv(bt2w_path)
        df_combined = attach_barttorvik_2week(df_combined, df_bt2w)
    if bthot_path.exists():
        df_hot      = pd.read_csv(bthot_path)
        df_combined = attach_barttorvik_hotness(df_combined, df_hot)
    df_combined = reorder_columns(df_combined)
    return df_combined


def compile_bracket_year(data_root: Path, year: int, round1_only: bool = False) -> None:
    """Write BracketCombinedData CSVs for all available rounds of a single year.

    Pass round1_only=True for the current year when the tournament has not yet
    been played – only Round 1 (the pre-tournament matchups) will be compiled.
    """
    kp_path    = data_root / 'Data' / 'KenPomData'            / f'{year}.csv'
    bt_path    = data_root / 'Data' / 'BartTorvikData'        / f'{year}.csv'
    bt2w_path  = data_root / 'Data' / '2WeekBartTorivkData'   / f'{year}.csv'
    bthot_path = data_root / 'Data' / 'HotnessBartTorvikData' / f'{year}.csv'
    df_kp      = pd.read_csv(kp_path)
    df_bt      = pd.read_csv(bt_path)    if bt_path.exists()    else None
    df_bt2w    = pd.read_csv(bt2w_path)  if bt2w_path.exists()  else None
    df_hot     = pd.read_csv(bthot_path) if bthot_path.exists() else None

    out_dir = data_root / 'Data' / 'BracketCombinedData' / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds = [1] if round1_only else BRACKET_ROUNDS
    for rnd in rounds:
        bracket_path = (
            data_root / 'Data' / 'BracketData' / str(year) / f'Round{rnd}_{year}.csv'
        )
        if not bracket_path.exists():
            print(f'    Round {rnd}: source file not found, skipping')
            continue

        df_round    = pd.read_csv(bracket_path)
        df_round    = rename_matchup_cols(df_round)
        df_combined = attach_kenpom(df_round, df_kp)
        if df_bt is not None:
            df_combined = attach_barttorvik(df_combined, df_bt)
        if df_bt2w is not None:
            df_combined = attach_barttorvik_2week(df_combined, df_bt2w)
        if df_hot is not None:
            df_combined = attach_barttorvik_hotness(df_combined, df_hot)
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
    parser.add_argument(
        '--this-year',
        type=int,
        default=None,
        help=(
            'Mark this year as the current (pre-tournament) year.  '
            'Only Round 1 of BracketCombinedData will be compiled for it, '
            'and it will be excluded from GameCombinedData.'
        ),
    )
    args = parser.parse_args()

    data_root  = Path(__file__).resolve().parents[1]
    this_year  = args.this_year

    # Build the year list; if --this-year is supplied and not already in ALL_YEARS,
    # append it so it gets processed.
    if args.year:
        years = [args.year]
    else:
        years = list(ALL_YEARS)
        if this_year is not None and this_year not in years:
            years.append(this_year)

    do_game    = args.type in ('game', 'both')
    do_bracket = args.type in ('bracket', 'both')

    # -----------------------------------------------------------------------
    # GameCombinedData
    # -----------------------------------------------------------------------
    if do_game:
        print('=== Compiling GameCombinedData ===')
        df_all = pd.DataFrame()
        for year in years:
            if this_year is not None and year == this_year:
                print(f'  {year}: skipped (--this-year; tournament not yet played)')
                continue
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

            round1_only = (this_year is not None and year == this_year)
            if round1_only:
                print(f'  {year}: (current year – Round 1 only)')
            else:
                print(f'  {year}:')
            compile_bracket_year(data_root, year, round1_only=round1_only)


if __name__ == '__main__':
    main()
