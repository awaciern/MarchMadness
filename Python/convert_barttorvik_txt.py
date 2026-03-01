"""
convert_barttorvik_txt.py

Converts a Bart Torvik .txt file (copied from barttorvik.com) to a flat CSV.

The raw .txt has an unusual multi-line format per team:

  Tournament teams (lines A-B form the header, then 20 stat lines):
    A: "Rk<TAB>TeamName"
    B: "   Seed seed, Round<TAB>Conf<TAB>G<TAB>W-L"
    C: "ConfRec<TAB>AdjOE"
    D: "Rk_AdjOE<TAB>AdjDE"
    ... (interleaved value / rank pairs for each stat)

  Non-tournament teams (1-line header, then 20 stat lines):
    A: "Rk<TAB>TeamName<TAB>Conf<TAB>G<TAB>W-L"
    B: "ConfRec<TAB>AdjOE"
    ...

Usage:
    python convert_barttorvik_txt.py <input_txt> <output_csv>
    python convert_barttorvik_txt.py <input_txt> <output_csv> --no-seeds
    python convert_barttorvik_txt.py <input_txt> <output_csv> --all-teams

Options:
    --no-seeds    Include all teams; leave Seed blank (use pre-tournament).
    --all-teams   Include non-tournament teams as well (implies --no-seeds for them).
                  By default only tournament-seeded teams are written.
"""

import argparse
import csv
import re
from pathlib import Path

HEADINGS = [
    'Rk', 'Team', 'Seed', 'Conf', 'G', 'Rec', 'ConfRec',
    'Wins', 'Losses', 'WinPct',
    'ConfWins', 'ConfLosses', 'ConfWinPct',
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

# Stat tokens after the team header, in order:
# each pair is (value, rank) except ConfRec which has no rank.
# Total of 41 tokens: ConfRec + 20×(value, rank).
STAT_SEQUENCE = [
    'ConfRec',
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


def is_team_start(line: str) -> bool:
    """A team-start line begins with an integer rank followed by a tab and then a letter."""
    return bool(re.match(r'^\d+\t[A-Za-z]', line))


def split_record(rec: str):
    """Split 'W–L' or 'W-L' into (wins, losses, winpct)."""
    rec = rec.replace('–', '-')
    w, l = rec.split('-')
    w, l = int(w), int(l)
    return w, l, round(w / (w + l), 4) if (w + l) > 0 else 0.0


def parse_block(block: list, no_seeds: bool, all_teams: bool):
    """
    Parse one team block (list of raw text lines) into a row dict.
    Returns None if the block should be skipped.
    """
    if not block:
        return None

    first = block[0]
    first_cols = first.split('\t')

    # Determine if this is a tournament/AQ team (2-line header) or a plain team (1-line header).
    # Matches old "   1 seed, Finals\t..." format AND new 2026 "   (A) 36 Ohio St.\t..." format.
    is_tournament = (
        len(block) > 1 and block[1][:1] == ' '
    )

    if is_tournament:
        rk = int(first_cols[0])
        team_name = first_cols[1].strip()

        # Parse seed line: "   N seed, Round\tConf\tG\tW-L"
        seed_line = block[1].strip()
        seed_parts = seed_line.split('\t')
        seed_info = seed_parts[0].strip()          # e.g. "1 seed, Finals"
        seed_match = re.match(r'^(\d+)\s+seed', seed_info)
        seed = int(seed_match.group(1)) if seed_match else ''
        conf = seed_parts[1] if len(seed_parts) > 1 else ''
        g    = int(seed_parts[2]) if len(seed_parts) > 2 else ''
        rec  = seed_parts[3].replace('–', '-') if len(seed_parts) > 3 else ''

        stat_lines = block[2:]
    else:
        # Non-tournament: all basic info on first line.
        if not all_teams and not no_seeds:
            return None  # skip non-tournament teams by default

        rk        = int(first_cols[0])
        team_name = first_cols[1].strip()
        conf      = first_cols[2].strip() if len(first_cols) > 2 else ''
        g         = int(first_cols[3]) if len(first_cols) > 3 else ''
        rec       = first_cols[4].replace('–', '-').strip() if len(first_cols) > 4 else ''
        seed      = ''

        stat_lines = block[1:]

    if no_seeds:
        seed = ''

    # Flatten stat tokens from the remaining lines.
    # Skip column-header repeat lines that appear at page breaks (e.g. "Rk\tTeam\tConf\t...").
    tokens = []
    for line in stat_lines:
        if re.match(r'^Rk\t', line):
            continue  # column header repeated at page break — skip
        for tok in line.split('\t'):
            tok = tok.strip()
            if tok:
                tokens.append(tok)

    # 2026+ format dropped ConfRec from the stat block (38 tokens instead of 39).
    # Detect this by checking whether the first token looks like a W-L record or a float.
    if tokens and '-' not in tokens[0] and not tokens[0].startswith('+'):
        # First token is a numeric value (AdjO), not a W-L record — ConfRec absent.
        tokens.insert(0, '')  # placeholder so STAT_SEQUENCE indices stay aligned

    if len(tokens) < len(STAT_SEQUENCE):
        return None  # malformed block

    stat_map = {}
    for i, name in enumerate(STAT_SEQUENCE):
        stat_map[name] = tokens[i]

    conf_rec = stat_map['ConfRec'].replace('–', '-')

    # Parse W-L
    wins = losses = winpct = ''
    if rec and '-' in rec:
        wins, losses, winpct = split_record(rec)

    conf_wins = conf_losses = conf_winpct = ''
    if conf_rec and '-' in conf_rec:
        conf_wins, conf_losses, conf_winpct = split_record(conf_rec)

    row = {
        'Rk': rk,
        'Team': team_name,
        'Seed': seed,
        'Conf': conf,
        'G': g,
        'Rec': rec,
        'ConfRec': conf_rec,
        'Wins': wins,
        'Losses': losses,
        'WinPct': winpct,
        'ConfWins': conf_wins,
        'ConfLosses': conf_losses,
        'ConfWinPct': conf_winpct,
    }

    # Numeric stats — ranks are int, values are float.
    rank_cols = {s for s in STAT_SEQUENCE if s.startswith('Rk_')}
    for name in STAT_SEQUENCE[1:]:  # skip ConfRec
        raw = stat_map[name]
        try:
            row[name] = int(raw) if name in rank_cols else float(raw.lstrip('+'))
        except ValueError:
            row[name] = raw

    return row


def main():
    parser = argparse.ArgumentParser(
        description='Convert a Bart Torvik .txt file to CSV.'
    )
    parser.add_argument('input_txt', help='Path to the input .txt file.')
    parser.add_argument('output_csv', help='Path for the output .csv file.')
    parser.add_argument(
        '--no-seeds', action='store_true',
        help='Include all teams; leave Seed blank (use when seeds are not yet assigned).',
    )
    parser.add_argument(
        '--all-teams', action='store_true',
        help='Include non-tournament teams as well (Seed left blank for them).',
    )
    args = parser.parse_args()

    no_seeds  = args.no_seeds
    all_teams = args.all_teams or no_seeds

    with open(args.input_txt, encoding='utf-8') as f:
        lines = f.read().splitlines()

    # Group lines into per-team blocks (skip header row).
    blocks: list = []
    current: list = []
    for line in lines[1:]:
        if is_team_start(line):
            if current:
                blocks.append(current)
            current = [line]
        else:
            if current:
                current.append(line)
    if current:
        blocks.append(current)

    rows = []
    skipped = 0
    for block in blocks:
        row = parse_block(block, no_seeds=no_seeds, all_teams=all_teams)
        if row is None:
            skipped += 1
        else:
            rows.append(row)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADINGS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Written {len(rows)} teams to {args.output_csv} ({skipped} skipped).')


if __name__ == '__main__':
    main()
