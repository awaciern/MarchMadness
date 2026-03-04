"""
find_name_mismatch.py

Diagnoses team name mismatches between BracketData Round 1 and the two stats
sources (KenPomData and BartTorvikData) for a given year.

Usage:
    python3 find_name_mismatch.py --year 2026
    python3 find_name_mismatch.py --year 2025
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / 'Data'


def bracket_teams(year: int) -> set:
    path = DATA_ROOT / 'BracketData' / str(year) / f'Round1_{year}.csv'
    if not path.exists():
        print(f'  [!] BracketData not found: {path}')
        return set()
    df = pd.read_csv(path)
    return set(df['Team1'].dropna()) | set(df['Team2'].dropna())


def stats_teams(subdir: str, year: int) -> set:
    path = DATA_ROOT / subdir / f'{year}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return set(df['Team'].dropna())


def report_mismatches(label: str, bracket: set, stats: set):
    in_bracket_not_stats = sorted(bracket - stats)
    in_stats_not_bracket = sorted(stats - bracket)
    print(f'\n--- {label} ---')
    print(f'  In BracketData but NOT in {label} ({len(in_bracket_not_stats)}):')
    for t in in_bracket_not_stats:
        print(f'    {t}')
    print(f'  In {label} but NOT in BracketData ({len(in_stats_not_bracket)}):')
    for t in in_stats_not_bracket:
        print(f'    {t}')


def main():
    parser = argparse.ArgumentParser(
        description='Find team name mismatches between BracketData and stats sources.'
    )
    parser.add_argument('--year', '-y', type=int, required=True, help='Tournament year.')
    args = parser.parse_args()
    year = args.year

    print(f'=== Name mismatch check for {year} ===')

    bracket = bracket_teams(year)
    if not bracket:
        return

    print(f'  BracketData Round 1: {len(bracket)} teams')

    kenpom = stats_teams('KenPomData', year)
    if kenpom is None:
        print(f'  [!] KenPomData/{year}.csv not found — skipping.')
    else:
        print(f'  KenPomData: {len(kenpom)} teams')
        report_mismatches('KenPomData', bracket, kenpom)

    barttorvik = stats_teams('BartTorvikData', year)
    if barttorvik is None:
        print(f'  [!] BartTorvikData/{year}.csv not found — skipping.')
    else:
        print(f'  BartTorvikData: {len(barttorvik)} teams')
        report_mismatches('BartTorvikData', bracket, barttorvik)

    barttorvik_2w = stats_teams('2WeekBartTorivkData', year)
    if barttorvik_2w is None:
        print(f'  [!] 2WeekBartTorivkData/{year}.csv not found — skipping.')
    else:
        print(f'  2WeekBartTorivkData: {len(barttorvik_2w)} teams')
        report_mismatches('2WeekBartTorivkData', bracket, barttorvik_2w)


if __name__ == '__main__':
    main()
