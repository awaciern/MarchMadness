"""
convert_evanmiya_txt.py

Converts an Evan Miya .txt file (copied from evanmiya.com) to a flat CSV.

The raw .txt has a simple vertical format:
  - Lines 1-21: column headers (one per line)
  - Then repeating 21-line blocks per team:
      Line  1: Rank (integer)
      Line  2: Team name (may include emoji badges like 🔥 🔒)
      Lines 3–21: numeric stat values in the same order as headers 3–21

The 21 columns in order:
  Relative Ranking, Team, O-Rate, D-Rate, Relative Rating,
  Opponent Adjust, Pace Adjust, Off Rank, Def Rank,
  True Tempo, Tempo Rank, Injury Rank, Home Rank, Roster Rank,
  Kill Shots Per Game, Kill Shots Conceded Per Game,
  Kill Shots Margin Per Game, Total Kill Shots,
  Total Kill Shots Conceded, D1 Wins, D1 Losses

Usage:
    python convert_evanmiya_txt.py <input_txt> <output_csv>

Example:
    python convert_evanmiya_txt.py Data/EvanMiyaData/2026.txt Data/EvanMiyaData/2026.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# Clean CSV column names (parallel to the 21 raw header lines).
HEADINGS = [
    'Rk',
    'Team',
    'ORate',
    'DRate',
    'Rating',
    'OppAdj',
    'PaceAdj',
    'OffRk',
    'DefRk',
    'Tempo',
    'TempoRk',
    'InjuryRk',
    'HomeRk',
    'RosterRk',
    'KSPerGame',
    'KSConcededPerGame',
    'KSMarginPerGame',
    'TotalKS',
    'TotalKSConceded',
    'Wins',
    'Losses',
]

# Derived columns appended after the raw ones.
EXTRA_HEADINGS = ['WinPct']

N_FIELDS = len(HEADINGS)  # 21 lines per block

# Integer columns (all "Rank" columns plus Wins/Losses/TotalKS/TotalKSConceded).
INT_COLS = {
    'Rk', 'OffRk', 'DefRk', 'TempoRk', 'InjuryRk', 'HomeRk', 'RosterRk',
    'TotalKS', 'TotalKSConceded', 'Wins', 'Losses',
}

# Regex to strip emoji and other non-ASCII characters from team names.
_NON_ASCII = re.compile(r'[^\x00-\x7F]')


def clean_team_name(raw: str) -> str:
    return _NON_ASCII.sub('', raw).strip()


def parse_value(col: str, raw: str):
    """Convert a raw string to int, float, or leave as string."""
    raw = raw.strip().lstrip('+')
    if not raw:
        return ''
    if col in INT_COLS:
        try:
            return int(float(raw))
        except ValueError:
            return raw
    try:
        return float(raw)
    except ValueError:
        return raw


def main():
    parser = argparse.ArgumentParser(
        description='Convert an Evan Miya .txt file to CSV.'
    )
    parser.add_argument('input_txt', help='Path to the input .txt file.')
    parser.add_argument('output_csv', help='Path for the output .csv file.')
    args = parser.parse_args()

    src = Path(args.input_txt)
    dst = Path(args.output_csv)

    if not src.exists():
        print(f'Error: {src} not found.', file=sys.stderr)
        sys.exit(1)

    lines = src.read_text(encoding='utf-8').splitlines()

    # The first N_FIELDS lines are the column headers — skip them.
    if len(lines) < N_FIELDS:
        print('Error: file too short to contain headers.', file=sys.stderr)
        sys.exit(1)

    data_lines = lines[N_FIELDS:]

    rows = []
    skipped = 0
    idx = 0
    while idx + N_FIELDS <= len(data_lines):
        block = data_lines[idx: idx + N_FIELDS]
        idx += N_FIELDS

        # Validate: first line should be an integer rank.
        try:
            rk = int(block[0].strip())
        except ValueError:
            skipped += 1
            continue

        team = clean_team_name(block[1])
        if not team:
            skipped += 1
            continue

        row = {'Rk': rk, 'Team': team}
        for col, raw in zip(HEADINGS[2:], block[2:]):
            row[col] = parse_value(col, raw)

        # Derived WinPct.
        wins   = row.get('Wins',   '')
        losses = row.get('Losses', '')
        if isinstance(wins, (int, float)) and isinstance(losses, (int, float)):
            total = wins + losses
            row['WinPct'] = round(wins / total, 4) if total > 0 else 0.0
        else:
            row['WinPct'] = ''

        rows.append(row)

    # Handle a trailing partial block (last team in file may be incomplete).
    remainder = data_lines[idx:]
    if remainder:
        try:
            rk = int(remainder[0].strip())
            team = clean_team_name(remainder[1]) if len(remainder) > 1 else ''
            if team:
                row = {'Rk': rk, 'Team': team}
                for col, raw in zip(HEADINGS[2:], remainder[2:]):
                    row[col] = parse_value(col, raw)
                # Fill any missing tail columns with empty string.
                for col in HEADINGS[2:]:
                    row.setdefault(col, '')
                wins   = row.get('Wins',   '')
                losses = row.get('Losses', '')
                if isinstance(wins, (int, float)) and isinstance(losses, (int, float)):
                    total = wins + losses
                    row['WinPct'] = round(wins / total, 4) if total > 0 else 0.0
                else:
                    row['WinPct'] = ''
                rows.append(row)
        except (ValueError, IndexError):
            skipped += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    all_cols = HEADINGS + EXTRA_HEADINGS
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f'Written {len(rows)} teams to {dst}  ({skipped} block(s) skipped).')


if __name__ == '__main__':
    main()
