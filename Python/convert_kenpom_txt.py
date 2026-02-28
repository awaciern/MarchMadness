"""
Converts a KenPom .txt file (tab-separated, seed embedded in team name)
to the same CSV format used by the KenPomData CSV files.

Usage: python convert_kenpom_txt.py <input_txt> <output_csv>
Example: python convert_kenpom_txt.py ../Data/KenPomData/2025.txt ../Data/KenPomData/2025.csv
"""

import csv
import sys

HEADINGS = [
    'Rk_AdjEM', 'Team', 'Seed', 'Conf', 'W-L', 'Wins', 'Losses', 'WinPct',
    'AdjEM', 'AdjO', 'Rk_AdjO', 'AdjD', 'Rk_AdjD',
    'AdjT', 'Rk_AdjT',
    'Luck', 'Rk_Luck',
    'SOS_AdjEM', 'Rk_SOS_AdjEM', 'SOS_AdjO', 'Rk_SOS_AdjO', 'SOS_AdjD', 'Rk_SOS_AdjD',
    'NCSOS_AdjEM', 'Rk_NCSOS_AdjEM',
]

def parse_float(val):
    """Strip leading + and convert to float, handling bare decimal forms like -.026"""
    return float(val)

def parse_row(cols):
    """
    Expected tab-split column positions in the txt file:
    0:  Rk
    1:  Team + embedded seed  (e.g. "Duke 1" or "Michigan St. 2")
    2:  Conf
    3:  W-L
    4:  AdjEM         (NetRtg, may have leading +)
    5:  AdjO          (ORtg)
    6:  Rk_AdjO
    7:  AdjD          (DRtg)
    8:  Rk_AdjD
    9:  AdjT
    10: Rk_AdjT
    11: Luck          (may have leading + or bare decimal like -.026)
    12: Rk_Luck
    13: SOS_AdjEM
    14: Rk_SOS_AdjEM
    15: SOS_AdjO
    16: Rk_SOS_AdjO
    17: SOS_AdjD
    18: Rk_SOS_AdjD
    19: NCSOS_AdjEM
    20: Rk_NCSOS_AdjEM
    """
    if len(cols) < 21:
        return None

    # Extract seed from end of team field
    team_field = cols[1]
    parts = team_field.rsplit(' ', 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None  # no seed — skip this team
    team_name = parts[0]
    seed = int(parts[1])

    wl = cols[3]
    wins, losses = wl.split('-')
    wins, losses = int(wins), int(losses)
    win_pct = wins / (wins + losses)

    return [
        int(cols[0]),                   # Rk_AdjEM
        team_name,                      # Team
        seed,                           # Seed
        cols[2],                        # Conf
        wl,                             # W-L
        wins,                           # Wins
        losses,                         # Losses
        win_pct,                        # WinPct
        parse_float(cols[4]),           # AdjEM
        parse_float(cols[5]),           # AdjO
        int(cols[6]),                   # Rk_AdjO
        parse_float(cols[7]),           # AdjD
        int(cols[8]),                   # Rk_AdjD
        parse_float(cols[9]),           # AdjT
        int(cols[10]),                  # Rk_AdjT
        parse_float(cols[11]),          # Luck
        int(cols[12]),                  # Rk_Luck
        parse_float(cols[13]),          # SOS_AdjEM
        int(cols[14]),                  # Rk_SOS_AdjEM
        parse_float(cols[15]),          # SOS_AdjO
        int(cols[16]),                  # Rk_SOS_AdjO
        parse_float(cols[17]),          # SOS_AdjD
        int(cols[18]),                  # Rk_SOS_AdjD
        parse_float(cols[19]),          # NCSOS_AdjEM
        int(cols[20]),                  # Rk_NCSOS_AdjEM
    ]

if len(sys.argv) != 3:
    print("Usage: python convert_kenpom_txt.py <input_txt> <output_csv>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

rows = []
with open(input_path, 'r') as f:
    for line in f:
        line = line.rstrip('\n')
        cols = line.split('\t')
        # Skip header/separator lines — first column must be a plain integer rank
        if not cols[0].strip().isdigit():
            continue
        row = parse_row(cols)
        if row is not None:
            rows.append(row)

with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(HEADINGS)
    writer.writerows(rows)

print(f"Wrote {len(rows)} teams to {output_path}")
