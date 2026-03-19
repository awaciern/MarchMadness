"""
parse_bracket_txt.py

Parse a .txt file copied from a PDF of a filled-out ESPN Men's Tournament
Challenge bracket and produce the bracket JSON format used in the Brackets/
folder.

The .txt file format is the layout produced by copying from the ESPN PDF:
  - Header lines (dates, round labels, point values)
  - EAST  section: 16 team lines  (seed name)
  - SOUTH section: 16 team lines
  - A central "picks" block containing R2 + S16 + E8 + FF picks interleaved
    in the visual reading order of the printed bracket
  - WEST   section: 16 team lines
  - MIDWEST section: 16 team lines

The parser structure in the central block (empirically determined):
  Tokens  0- 7:  R2  East   picks (8)
  Tokens  8-15:  R2  South  picks (8)
  Tokens 16-17:  S16 East   picks (2, inner games closest to center)
  Tokens 18-19:  S16 West   picks (2, inner games)
  Tokens 20-21:  E8  East   picks (2)
  Tokens 22-23:  E8  West   picks (2)
  Tokens 24-25:  S16 South  picks (2, inner games)
  Tokens 26-27:  S16 Midwest picks (2, inner games)
  [FF / Champion area — variable, scanned separately]
  Tokens ~33-36: S16 South  remaining picks (2 more)
  Tokens ~37-40: S16 Midwest remaining picks (2 more)
  Tokens ~41-42: E8 South / Midwest signals
  ...
  Tokens ~45-52: S16 West  picks (4), R2 West partial
  Tokens ~53-60: R2 Midwest picks (8)

Because the interleaving is complex, the parser uses a greedy
approach: the R2 picks (clearly positioned at tokens 0-15 and ~53-60)
are extracted first and used to build the valid-teams pool for each
subsequent round, then each remaining token is matched greedily to
the earliest-round slot it can fit.

Usage (standalone)
------------------
    python3 Python/parse_bracket_txt.py \\
        --txt  Brackets/Text/DefendingChamp.txt \\
        --name "Defending Champ" \\
        --group "ESPN Challenge" \\
        [--year 2026] \\
        [--out  Brackets/ESPN_Challenge/Defending_Champ.json]

    # Dry-run (print JSON to stdout, don't write file)
    python3 Python/parse_bracket_txt.py \\
        --txt Brackets/Text/DefendingChamp.txt \\
        --name "Test" --group "Test" --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

# Abbreviations / alternate spellings that appear in the PDF text but differ
# from the canonical names used in Round1_<year>.csv.
# Keys are lowercase+stripped versions of what the PDF produces;
# values are the canonical CSV name.
_ABBREV_MAP: dict[str, str] = {
    # Common short forms seen in ESPN bracket PDFs
    'st john\'s':       "St. John's",
    'st johns':         "St. John's",
    'st. john\'s':      "St. John's",
    'michigan st':      'Michigan State',
    'michigan state':   'Michigan State',
    'ohio state':       'Ohio State',
    'iowa state':       'Iowa State',
    'n dakota st':      'North Dakota State',
    'n. dakota st':     'North Dakota State',
    'north dakota st':  'North Dakota State',
    'north dakota state': 'North Dakota State',
    'ca baptist':       'Cal Baptist',
    'cal baptist':      'Cal Baptist',
    'north carolina':   'North Carolina',
    'south florida':    'South Florida',
    'prairie view':     'Prairie View A&M',
    'prairie view a&m': 'Prairie View A&M',
    'ucf':              'UCF',
    'uconn':            'UConn',
    'vcu':              'VCU',
    'byu':              'BYU',
    'miami (fl)':       'Miami (FL)',
    'miami fl':         'Miami (FL)',
    'miami':            'Miami (FL)',   # resolved after context check
    'miami oh':         'Miami OH',
    'miami (oh)':       'Miami OH',
    'kennesaw st':      'Kennesaw State',
    'kennesaw state':   'Kennesaw State',
    'long island':      'Long Island',
    'high point':       'High Point',
    "hawai'i":          'Hawaii',
    'hawaii':           'Hawaii',
    'wright st':        'Wright State',
    'wright state':     'Wright State',
    'santa clara':      'Santa Clara',
    'saint mary\'s':    "Saint Mary's",
    'saint louis':      'Saint Louis',
    'saint louis':      'Saint Louis',
    'tennessee st.':    'Tennessee St.',
    'tennessee st':     'Tennessee St.',
    'tennessee state':  'Tennessee St.',
    'n. dakota st.':    'North Dakota State',
    'texas a&m':        'Texas A&M',
}


def _normalise_key(s: str) -> str:
    """Lower-case, strip, collapse whitespace."""
    return re.sub(r'\s+', ' ', s.strip().lower())


def build_name_lookup(r1_teams: list[str]) -> dict[str, str]:
    """
    Build a lookup from every reasonable normalised variant of a team name
    to its canonical CSV name.

    Priority order:
      1. Exact (normalised) match
      2. Abbreviation map entry
      3. First-word match (for cases like 'Michigan' → 'Michigan' vs
         'Michigan State' — the longer match wins if both fit)
    """
    lookup: dict[str, str] = {}

    # First pass: exact normalised match
    for team in r1_teams:
        k = _normalise_key(team)
        lookup[k] = team

    # Second pass: abbreviation overrides
    for abbrev, canonical in _ABBREV_MAP.items():
        # Only apply if canonical is actually in this year's bracket
        if canonical in r1_teams:
            lookup[_normalise_key(abbrev)] = canonical

    return lookup


def resolve_team(token: str, lookup: dict[str, str]) -> Optional[str]:
    """
    Try to resolve a text token to a canonical team name.

    Returns the canonical name or None if no match found.
    """
    t = _normalise_key(token)
    if t in lookup:
        return lookup[t]

    # Try stripping a leading seed number (e.g. "1 Duke" → "Duke")
    m = re.match(r'^\d{1,2}\s+(.+)$', t)
    if m:
        name_part = m.group(1).strip()
        if name_part in lookup:
            return lookup[name_part]

    return None


# ---------------------------------------------------------------------------
# Text file parser
# ---------------------------------------------------------------------------

# Lines that are purely structural/noise (not team names or picks)
_NOISE_PATTERNS = [
    r'^\d{1,2}/\d{1,2}/\d{2,4}',      # date header
    r'^round of',
    r'^sweet 16',
    r'^elite 8',
    r'^final rounds?',
    r'^mar \d',
    r'^apr \d',
    r'^\d+ - \d+ pts',
    r'^\d+ pts',
    r'^my pick',
    r'^champion$',
    r'^presented by',
    r'^indianapolis',
    r'^how many total',
    r'^will be scored',
    r'^championship game',
    r'^wolverines$',
    r'^jayhawks$',
    r'^hoosiers$',
    r'^dukies$',              # mascots
    r'^tar heels$',
    r'^\d+$',                 # lone numbers (tiebreaker score)
    r'^[a-z ]+ in$',         # "Indianapolis, IN"
    r'^,',
]


def _is_noise(line: str) -> bool:
    lo = line.lower().strip()
    for pat in _NOISE_PATTERNS:
        if re.match(pat, lo):
            return True
    # Mascot lines — all letters, no digits, and not a known team
    if re.match(r'^[a-z ]+$', lo) and len(lo.split()) <= 2 and lo not in (
        'duke', 'byu', 'vcu', 'ucf', 'penn', 'iowa', 'troy', 'ohio', 'hawaii',
        'texas', 'purdue', 'kansas', 'miami',
    ):
        # Could be a mascot — we'll handle this by only accepting lines that
        # match a known team
        pass
    return False


def _strip_seed(line: str) -> str:
    """Remove a leading seed number from a line like '1 Duke' → 'Duke'."""
    return re.sub(r'^\d{1,2}\s+', '', line.strip())


def _parse_region_block(lines: list[str], start: int) -> list[tuple[str, str]]:
    """
    Parse 16 team lines starting at `start` (after the region header).
    Returns list of (team1, team2) for the 8 matchups (seed order: 1v16, 8v9, …).
    """
    teams = []
    i = start
    while i < len(lines) and len(teams) < 16:
        line = lines[i].strip()
        if re.match(r'^\d{1,2}\s+\S', line):
            teams.append(_strip_seed(line))
        elif line and not _is_noise(line) and not re.match(
            r'^(EAST|WEST|SOUTH|MIDWEST|ROUND|SWEET|ELITE|FINAL|MAR|APR)',
            line.upper(),
        ):
            # Non-seeded but non-noise — skip it. Some PDFs omit seed here.
            pass
        i += 1
    if len(teams) < 16:
        raise ValueError(
            f'Expected 16 teams in region block starting at line {start}; '
            f'got {len(teams)}: {teams}'
        )
    # Pair them up: 1&2, 3&4, etc.
    matchups = [(teams[k * 2], teams[k * 2 + 1]) for k in range(8)]
    return matchups


def _find_region(lines: list[str], region_name: str) -> int:
    """Return index of the REGION header line (case-insensitive)."""
    for i, l in enumerate(lines):
        if l.strip().upper() == region_name.upper():
            return i
    raise ValueError(f"Region '{region_name}' not found in text file.")


def parse_txt(txt_path: Path) -> dict:
    """
    Parse the bracket PDF text file and return a dict with:
        {
          'r1': list[str],    # 32 winners (each game's picked winner)
          'r2': list[str],    # 16
          's16': list[str],   # 8
          'e8': list[str],    # 4
          'semi': list[str],  # 2
          'champion': str,
        }

    The team order in each list matches the Brackets JSON convention used by
    the app:
        r1:  South[0-7], East[8-15], Midwest[16-23], West[24-31]
        r2:  South[0-3], East[4-7], Midwest[8-11], West[12-15]
        s16: South[0-1], East[2-3], Midwest[4-5], West[6-7]
        e8:  South[0],   East[1],   Midwest[2],   West[3]
        semi: pair0-winner, pair1-winner
    """
    text = txt_path.read_text(encoding='utf-8', errors='replace')
    raw_lines = text.splitlines()
    lines = [l.strip() for l in raw_lines]

    # ---- 1. Extract the four R1 regions ----
    east_idx     = _find_region(lines, 'EAST')
    south_idx    = _find_region(lines, 'SOUTH')
    west_idx     = _find_region(lines, 'WEST')
    midwest_idx  = _find_region(lines, 'MIDWEST')

    east_matchups    = _parse_region_block(lines, east_idx + 1)
    south_matchups   = _parse_region_block(lines, south_idx + 1)
    west_matchups    = _parse_region_block(lines, west_idx + 1)
    midwest_matchups = _parse_region_block(lines, midwest_idx + 1)

    # Canonical CSV order: South, East, Midwest, West
    all_matchups = south_matchups + east_matchups + midwest_matchups + west_matchups
    # all teams that appear in R1
    all_teams: list[str] = []
    for t1, t2 in all_matchups:
        all_teams.append(t1)
        all_teams.append(t2)

    lookup = build_name_lookup(all_teams)

    # ---- 2. Resolve all tokens in the "picks" block ----
    # The picks block occupies the lines between the end of SOUTH's 16 teams and
    # the beginning of the WEST region header (the first of the 4 regions listed
    # last in the text).  EAST and SOUTH appear first; WEST and MIDWEST appear
    # at the bottom.  The central block therefore runs from after SOUTH's last
    # team to before WEST (or MIDWEST, whichever comes first).
    south_end  = south_idx + 1 + 16  # approximate end of south block
    picks_end  = min(west_idx, midwest_idx)

    central_tokens: list[str] = []
    for i in range(south_end, picks_end):
        line = lines[i].strip()
        if not line:
            continue
        # Skip pure noise
        if _is_noise(line):
            continue
        # Collect the line — it may be "N Team" or just "Team"
        team = resolve_team(line, lookup)
        if team is not None:
            central_tokens.append(team)
        # else: skip non-team lines (venue, tiebreaker, "Presented By", etc.)

    # ---- 3. Map tokens → rounds ----
    # The PDF picks block for a 64-team bracket has this token count per half:
    #   R2 East (8) + R2 South (8) = 16 tokens
    #   S16 East (4) + S16 South (4) = 8 tokens
    #   E8 East (2) + E8 South (2) = 4 tokens  [left half Total: 28 left side]
    #   FF picks (2) + Champion (1) = 3  ← but these appear mid-block
    #   E8 West (2) + E8 Midwest (2) = 4
    #   S16 West (4) + S16 Midwest (4) = 8
    #   R2 West (8) + R2 Midwest (8) = 16
    # Plus mascot/team pair for champion area
    # Total non-champion picks: 16+8+4+4+8+16 = 56 team tokens + champion = 57
    #
    # However the "My Pick" section introduces two Final Four team labels
    # followed by champion team + mascot line (skip mascot), so the champion
    # appears as the first resolved team after "Champion" in the noise check.
    #
    # Most robust approach: use the known matchup structure to validate each
    # pick against the possible teams at that stage, from the already-known
    # R1 bracket.  We simulate the bracket forward region-by-region.

    picks = _infer_picks(central_tokens, all_matchups, lookup, lines, picks_end)
    return picks


def _advance(pool: list[str], team: str, region_teams: list[str]) -> str:
    """
    Given a list of possible teams at a stage, return `team` if it is in the
    pool; otherwise return the first element of pool (fallback).
    """
    if team in pool:
        return team
    return pool[0] if pool else team


def _infer_picks(
    tokens: list[str],
    all_matchups: list[tuple[str, str]],
    lookup: dict[str, str],
    lines: list[str],
    picks_end_idx: int,
) -> dict:
    """
    The PDF lays the bracket out visually.  The central block (between SOUTH
    and WEST) lists picks in this order for the *left* half of the bracket
    (East + South), then the *right* half (Midwest + West):

    Left half (top half of the printed bracket, East on left, South on right):
      Block A: R2  East (8 picks):   East games 0-7  half-winners
      Block B: R2  South (8 picks):  South games 0-7 half-winners
      Block C: S16 East  (4 picks)  } interleaved as
      Block D: S16 South (4 picks)  } columns in the PDF, but in the text
      ...                           } they are dumped as one block of 8
      Block E: E8  East  (2 picks)  }
      Block F: E8  South (2 picks)  }
      --- then champion area (2 FF + 1 champ + extras) ---
      Block G: S16 Midwest (4 picks) 
      Block H: S16 West   (4 picks)
      Block I: R2  Midwest (8 picks)
      Block J: R2  West   (8 picks)

    (E8 Midwest + West are embedded mid-block in the champion area.)

    The actual champion often appears with a label ("Champion") in the noise
    lines, so we hunt for it there separately.

    To handle variations robustly we use a greedy forward-simulation approach:
    consume tokens in order, matching each token to the earliest round/region
    it can legally belong to given the bracket structure.
    """
    # Region indices for the four regions in CSV order
    # SOUTH=0, EAST=1, MIDWEST=2, WEST=3
    # Matchup indexing:
    #   all_matchups[0..7]   = South
    #   all_matchups[8..15]  = East
    #   all_matchups[16..23] = Midwest
    #   all_matchups[24..31] = West

    # ----- Build per-region R1 team sets -----
    def region_r1(region_idx: int) -> list[tuple[str, str]]:
        return all_matchups[region_idx * 8: region_idx * 8 + 8]

    # ----- The PDF text is structured: -----
    # EAST R1 ... SOUTH R1 ...
    # [central picks block]
    # WEST R1 ... MIDWEST R1 ...
    #
    # Central block token order (empirically determined from the ESPN PDF):
    #
    # Tokens 0-7:   R2 East     (8 picks)
    # Tokens 8-15:  R2 South    (8 picks)
    # Tokens 16-19: S16 East    (4 picks)
    # Tokens 20-23: S16 South   (4 picks)  -- may be merged with East
    # Tokens 24-25: E8 East     (2 picks)
    # Tokens 26-27: E8 South    (2 picks)
    # -- champion area (variable length) --
    # Tokens ?+0-3: E8 Midwest + E8 West (4 picks, mixed)
    # Tokens ?+4-7: S16 Midwest (4 picks)
    # Tokens ?+8-11:S16 West   (4 picks)
    # Tokens ?+12-19: R2 Midwest (8 picks)
    # Tokens ?+20-27: R2 West   (8 picks)
    #
    # Total non-champ picks = 8+8+8+4+4+8+8 = 48 + champion = 49 ... but the
    # interleaving varies slightly by PDF layout.
    #
    # Most reliable: split at the champion detection point.

    # First, find the champion from the raw lines (look for "Champion" label)
    champion: Optional[str] = _find_champion(lines, lookup)

    # Try to find Final Four teams from "My Pick" section
    ff_picks = _find_ff_picks(lines, lookup)

    # ----- Slice tokens into left and right halves -----
    # The champion area separates them.  We use the ff_picks to detect
    # the midpoint, otherwise fall back to index 28.
    left_tokens, right_tokens = _split_tokens_at_champion_area(
        tokens, ff_picks, champion
    )

    # ----- Assign tokens to rounds -----
    r2_east    = left_tokens[0:8]
    r2_south   = left_tokens[8:16]
    # S16 and E8 from left (16 more)
    left_mid   = left_tokens[16:]
    s16_east   = left_mid[0:4]
    s16_south  = left_mid[4:8]
    e8_east_south = left_mid[8:12]  # 2 East + 2 South

    right_mid  = right_tokens
    e8_midwest_west = right_mid[0:4]  # 2 Midwest + 2 West
    s16_south_right = right_mid[4:8]  # Sometimes Midwest S16 comes first
    s16_west        = right_mid[8:12]
    r2_midwest      = right_mid[12:20]
    r2_west         = right_mid[20:28]

    # E8 ordering in the PDF: appears to be alternating East/South then
    # Midwest/West, but the exact interleave depends on the bracket layout.
    # We pick the first token for each region's E8 slot.
    e8_east  = e8_east_south[0:1]
    e8_south = e8_east_south[1:2]
    e8_mw    = e8_midwest_west[0:2]  # Midwest
    e8_west  = e8_midwest_west[2:4]  # West

    def first(lst: list[str], fallback: str = '') -> str:
        return lst[0] if lst else fallback

    # ----- Validate each pick against the legal R1 bracket -----
    r2_south_valid = _validate_round(r2_south, region_r1(0), 'R2 South')
    r2_east_valid  = _validate_round(r2_east,  region_r1(1), 'R2 East')
    r2_mw_valid    = _validate_round(r2_midwest[:8], region_r1(2), 'R2 Midwest')
    r2_west_valid  = _validate_round(r2_west[:8],   region_r1(3), 'R2 West')

    # Build R1 picks from R2 picks (each R2 pick wins one R1 game)
    r1_picks = _r1_from_r2(
        r2_south_valid + r2_east_valid + r2_mw_valid + r2_west_valid,
        all_matchups,
    )

    # ---- Assemble R2 in CSV order (South, East, Midwest, West) ----
    r2_picks = r2_south_valid + r2_east_valid + r2_mw_valid + r2_west_valid

    # ---- S16 (South, East, Midwest, West) ----
    # From right block: right_mid[4:12] = S16 Midwest + S16 West (or reversed)
    # We match by checking which teams from R2 are present
    s16_south_valid = _validate_round_full(s16_south, r2_south_valid, 'S16 South', pairs=4)
    s16_east_valid  = _validate_round_full(s16_east,  r2_east_valid,  'S16 East',  pairs=4)
    s16_mw_valid    = _validate_round_full(s16_south_right[:4], r2_mw_valid, 'S16 Midwest', pairs=4)
    s16_west_valid  = _validate_round_full(s16_west[:4], r2_west_valid, 'S16 West', pairs=4)

    s16_picks = s16_south_valid + s16_east_valid + s16_mw_valid + s16_west_valid

    # ---- E8 (South, East, Midwest, West) ----
    # e8_east/south from left block; e8_mw/west from right block
    e8_south_v = _validate_round_full([first(e8_south)], s16_south_valid, 'E8 South',   pairs=2)
    e8_east_v  = _validate_round_full([first(e8_east)],  s16_east_valid,  'E8 East',    pairs=2)
    e8_mw_v    = _validate_round_full([first(e8_mw)],    s16_mw_valid,    'E8 Midwest', pairs=2)
    e8_west_v  = _validate_round_full([first(e8_west)],  s16_west_valid,  'E8 West',    pairs=2)

    e8_picks = e8_south_v + e8_east_v + e8_mw_v + e8_west_v

    # ---- Semi / Final Four ----
    semi_picks = list(ff_picks) if ff_picks and len(ff_picks) >= 2 else e8_picks[:2]

    # ---- Champion ----
    champ = champion or (semi_picks[0] if semi_picks else '')

    return {
        'r1':       r1_picks,
        'r2':       r2_picks,
        's16':      s16_picks,
        'e8':       e8_picks,
        'semi':     semi_picks,
        'champion': champ,
    }


def _find_champion(lines: list[str], lookup: dict[str, str]) -> Optional[str]:
    """
    Find the user's champion pick.

    ESPN bracket PDF layout in the champion area:
      "My Pick"
      <FF team 1>
      [<non-finalist label>]   ← sometimes a seed/mascot appears here
      <champion name>          ← the actual champion pick (before a mascot line)
      "Wolverines" / mascot
      "Champion"               ← structural label
      <runner-up>              ← first team listed after Champion label
      <champion>               ← second team listed after Champion label

    Strategy: after "Champion" label, collect up to 4 team tokens and return
    the LAST one (the champion is listed after the runner-up in the ESPN layout).
    If only one team is found there, return it.
    """
    for i, line in enumerate(lines):
        if re.match(r'^champion$', line.strip().lower()):
            candidates: list[str] = []
            for j in range(i + 1, min(i + 8, len(lines))):
                t = resolve_team(lines[j], lookup)
                if t:
                    candidates.append(t)
            if candidates:
                return candidates[-1]   # champion is listed LAST in this area
    return None


def _find_ff_picks(lines: list[str], lookup: dict[str, str]) -> list[str]:
    """
    Find the user's Final Four team picks from the 'My Pick' section.

    ESPN bracket PDF layout:
      "My Pick"
      <FF team 1>              ← seed label visible in that column
      [<non-finalist entry>]   ← sometimes a seed label bleeds in
      <champion name>
      <mascot>                 ← always follows the champion name
      "Champion"

    We collect up to 3 team tokens after "My Pick" and then discard the one
    that is immediately followed by a mascot/noise line (that's the champion,
    which will appear again after the "Champion" label).  The remaining two
    are the Final Four picks.

    Fallback: if we can't distinguish, return the first two teams found.
    """
    for i, line in enumerate(lines):
        if re.match(r'^my pick$', line.strip().lower()):
            # collect (team, next_line_is_mascot) pairs
            window_teams: list[tuple[str, bool]] = []
            j = i + 1
            while j < min(i + 12, len(lines)) and len(window_teams) < 4:
                raw = lines[j].strip()
                t = resolve_team(raw, lookup)
                if t:
                    # check if the very next line is a mascot/noise
                    next_is_noise = False
                    if j + 1 < len(lines):
                        nxt = lines[j + 1].strip().lower()
                        if re.match(r'^champion$', nxt) or nxt in {
                            'wolverines', 'jayhawks', 'hoosiers', 'cardinals',
                            'wildcats', 'longhorns', 'bulldogs', 'tigers',
                            'gators', 'tarheels', 'tar heels', 'trojans', 'bears',
                            'knights', 'eagles', 'hawks', 'huskies', 'zags',
                            'aztecs', 'cougars', 'cyclones', 'ducks', 'heels',
                            'wolfpack', 'seminoles', 'gamecocks', 'razorbacks',
                            'boilermakers', 'huskers', 'commodores', 'broncos',
                            'owls', 'rams', 'falcons', 'panthers',
                        }:
                            next_is_noise = True
                    window_teams.append((t, next_is_noise))
                j += 1
            # The champion entry is the one followed by a mascot/noise line.
            # The FF picks are the others.
            ff = [t for t, is_champ in window_teams if not is_champ]
            if len(ff) >= 2:
                return ff[:2]
            # Fallback: return first two team tokens
            return [t for t, _ in window_teams][:2]
    return []


def _split_tokens_at_champion_area(
    tokens: list[str],
    ff_picks: list[str],
    champion: Optional[str],
) -> tuple[list[str], list[str]]:
    """
    Split the central token list into left-half and right-half picks.
    The split point is located by finding where the FF picks appear in the
    token stream — they create a 'break' between the two halves.
    """
    total = len(tokens)
    if total == 0:
        return [], []

    # Heuristic split: the central block usually has 28+X tokens on each side.
    # We look for the midpoint around index 28, but adjust to avoid putting
    # FF picks on the wrong side.
    midpoint = min(28, total // 2 + 4)

    # If we have FF picks, find the latest occurrence before midpoint+6;
    # the split is just after that.
    if ff_picks:
        for sep in range(min(midpoint + 6, total - 1), 0, -1):
            if tokens[sep - 1] in ff_picks or (champion and tokens[sep - 1] == champion):
                # Found the champion area boundary
                return tokens[:sep], tokens[sep:]

    return tokens[:midpoint], tokens[midpoint:]


def _validate_round(
    picks: list[str],
    r1_matchups: list[tuple[str, str]],
    label: str,
) -> list[str]:
    """
    For each of 8 pick slots (R2 round), verify the pick is one of the two
    R1 teams for that game.  If the pick is invalid or missing, fall back to
    team1 of that matchup.
    Returns exactly 8 canonical team names.
    """
    result: list[str] = []
    for i, (t1, t2) in enumerate(r1_matchups):
        pick = picks[i] if i < len(picks) else None
        if pick == t1 or pick == t2:
            result.append(pick)
        else:
            # pick might be the name with slightly different form — check
            result.append(t1)  # fallback
    return result


def _validate_round_full(
    picks: list[str],
    prev_winners: list[str],
    label: str,
    pairs: int,
) -> list[str]:
    """
    For a round with `pairs` games, each winner must come from adjacent pairs
    of `prev_winners`.  Fill missing with prev_winners[game*2] (upper seed).
    """
    result: list[str] = []
    for i in range(pairs):
        pair_a = prev_winners[i * 2] if i * 2 < len(prev_winners) else ''
        pair_b = prev_winners[i * 2 + 1] if i * 2 + 1 < len(prev_winners) else ''
        pick = picks[i] if i < len(picks) else None
        if pick and (pick == pair_a or pick == pair_b):
            result.append(pick)
        else:
            result.append(pair_a)  # fallback: higher seed
    return result


def _r1_from_r2(r2_picks: list[str], all_matchups: list[tuple[str, str]]) -> list[str]:
    """
    Derive R1 picks: each R2 winner picks themselves as the winner of their
    R1 game.  This is always correct — the R2 winner IS the R1 winner.
    """
    # all_matchups is in CSV order (South, East, Midwest, West) × 8 games
    # r2_picks is in same order (32 → 16 reduces to 16 items)
    r1 = []
    for i, winner in enumerate(r2_picks):
        t1, t2 = all_matchups[i]
        if winner == t1 or winner == t2:
            r1.append(winner)
        else:
            r1.append(t1)  # fallback
    return r1


# ---------------------------------------------------------------------------
# Load Round1 CSV and build authoritative team list
# ---------------------------------------------------------------------------

def load_round1_csv(csv_path: Path) -> list[tuple[str, str]]:
    """
    Load the Round1_<year>.csv and return a list of (team1, team2) tuples
    in CSV order (32 matchups: South×8, East×8, Midwest×8, West×8).
    """
    matchups: list[tuple[str, str]] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            matchups.append((row['Team1'].strip(), row['Team2'].strip()))
    if len(matchups) != 32:
        raise ValueError(
            f'Expected 32 rows in {csv_path}; got {len(matchups)}'
        )
    return matchups


def txt_to_bracket_json(
    txt_path: Path,
    name: str,
    group: str,
    year: int,
    ff_pairings: str,
    csv_matchups: Optional[list[tuple[str, str]]] = None,
    data_root: Optional[Path] = None,
) -> dict:
    """
    High-level function: parse the txt, validate against the CSV bracket, and
    produce the complete bracket JSON dict.

    If `csv_matchups` is provided it is used for validation instead of loading
    the CSV from disk.  If neither is provided the csv is located from
    `data_root` or from the script's repo root.
    """
    import datetime

    # Load the authoritative matchup list if not provided
    if csv_matchups is None:
        root = data_root or Path(__file__).resolve().parents[1]
        csv_path = root / 'Data' / 'BracketData' / str(year) / f'Round1_{year}.csv'
        csv_matchups = load_round1_csv(csv_path)

    # Override the text-file's region extraction with the canonical CSV names
    picks = _parse_with_csv(txt_path, csv_matchups)

    return {
        'name':        name,
        'group':       group,
        'year':        year,
        'created':     datetime.datetime.now().isoformat(timespec='seconds'),
        'ff_pairings': ff_pairings,
        'picks':       picks,
    }


def _parse_with_csv(txt_path: Path, csv_matchups: list[tuple[str, str]]) -> dict:
    """
    Parse the txt file using the authoritative CSV matchups for validation.

    ESPN bracket PDF central block layout (empirically determined, 64-team):
      Tokens  0- 7 : R2 East  (8)          -- before "My Pick" label
      Tokens  8-15 : R2 South (8)
      Tokens 16..  : S16/E8 inner picks    -- interleaved East+West, before FF area
      ------ "My Pick" / "Champion" area (skipped) ------
      .. cont ..   : S16/E8 inner picks    -- South+Midwest, after Champion area
      Tokens [-16:-8]: R2 West  (8)        -- always 2nd-to-last 8 in whole block
      Tokens [-8:] : R2 Midwest (8)        -- always last 8 in block

    Approach:
      1. Collect tokens from the block in two passes: before the "My Pick" line
         and after the last label line related to FF/Champion.
      2. Concatenate the two passes → combined token list.
      3. R2 East = combined[0:8], R2 South = combined[8:16].
      4. R2 Midwest = combined[-8:], R2 West = combined[-16:-8].
      5. Inner picks = combined[16:-16], used greedily for S16/E8.
      6. Champion and semi extracted from labeled lines.
    """
    text = txt_path.read_text(encoding='utf-8', errors='replace')
    lines = [l.strip() for l in text.splitlines()]

    all_teams = [t for pair in csv_matchups for t in pair]
    lookup = build_name_lookup(all_teams)

    east_idx    = _find_region(lines, 'EAST')
    south_idx   = _find_region(lines, 'SOUTH')
    west_idx    = _find_region(lines, 'WEST')
    midwest_idx = _find_region(lines, 'MIDWEST')

    south_end = south_idx + 17
    picks_end = min(west_idx, midwest_idx)

    # ---- Collect tokens in two passes around the FF/Champion area ----
    # Pass 1: from south_end until the "My Pick" label line
    # Pass 2: after the last "Champion" / tiebreaker block until picks_end

    # Find the "My Pick" line index
    mypick_idx = None
    for i in range(south_end, picks_end):
        if re.match(r'^my pick$', lines[i].lower()):
            mypick_idx = i
            break

    # Find the last "noise" line in the champion area (after "Champion" label)
    champ_area_end = None
    if mypick_idx is not None:
        for i in range(mypick_idx, picks_end):
            lo = lines[i].lower()
            if re.search(r'championship game|how many total|will be scored|presented by', lo):
                # After these lines is where the right-half picks resume
                # Skip past them
                champ_area_end = i + 1
                break
        if champ_area_end is None:
            # Fallback: skip until "Champion" label, then skip 6 more lines
            for i in range(mypick_idx, picks_end):
                if re.match(r'^champion$', lines[i].lower()):
                    champ_area_end = i + 6
                    break
        if champ_area_end is None:
            champ_area_end = mypick_idx + 10

    def _collect_tokens(start: int, end: int) -> list[str]:
        toks = []
        for i in range(start, min(end, picks_end)):
            line = lines[i].strip()
            if not line or _is_noise(line):
                continue
            t = resolve_team(line, lookup)
            if t is not None:
                toks.append(t)
        return toks

    if mypick_idx is not None:
        pre_ff  = _collect_tokens(south_end, mypick_idx)
        post_ff = _collect_tokens(champ_area_end, picks_end)
        combined = pre_ff + post_ff
    else:
        combined = _collect_tokens(south_end, picks_end)

    # ---- R1 picks from fixed positions in combined ----
    # Tokens 0-7   = R1 East  (8 R1 east winners)
    # Tokens 8-15  = R1 South (8 R1 south winners)
    # Tokens -16:-8= R1 West  (8 R1 west winners)
    # Tokens -8:   = R1 Midwest (8 R1 midwest winners)
    n = len(combined)
    r1_east_raw    = combined[0:8]
    r1_south_raw   = combined[8:16]
    r1_west_raw    = combined[n - 16: n - 8] if n >= 32 else []
    r1_midwest_raw = combined[n - 8: n]      if n >= 24 else []

    south_mu   = csv_matchups[0:8]
    east_mu    = csv_matchups[8:16]
    midwest_mu = csv_matchups[16:24]
    west_mu    = csv_matchups[24:32]

    def _val_r1(raw, matchups):
        out = []
        for i, (t1, t2) in enumerate(matchups):
            p = raw[i] if i < len(raw) else None
            out.append(p if p in (t1, t2) else t1)
        return out

    r1_south_v = _val_r1(r1_south_raw, south_mu)
    r1_east_v  = _val_r1(r1_east_raw,  east_mu)
    r1_mw_v    = _val_r1(r1_midwest_raw, midwest_mu)
    r1_west_v  = _val_r1(r1_west_raw,  west_mu)

    # JSON r1 order: South + East + Midwest + West
    r1_picks = r1_south_v + r1_east_v + r1_mw_v + r1_west_v

    # ---- Inner tokens for R2/S16/E8 greedy matching ----
    inner = combined[16: n - 16] if n > 32 else []

    order = ('east', 'west', 'south', 'midwest')

    def _greedy(inner_toks, pools_init, n_per):
        pools = {k: list(v) for k, v in pools_init.items()}
        picks = {k: [] for k in pools_init}
        for t in inner_toks:
            for key in order:
                if t in pools[key] and len(picks[key]) < n_per:
                    picks[key].append(t)
                    pools[key].remove(t)
                    break
        return picks

    def _fill_to(lst, pool, n):
        out = list(lst)
        for t in pool:
            if len(out) >= n:
                break
            if t not in out:
                out.append(t)
        return out[:n]

    # ---- R2 greedy: 4 per region from R1 pool ----
    r2_by = _greedy(inner, {'south': r1_south_v, 'east': r1_east_v,
                             'midwest': r1_mw_v, 'west': r1_west_v}, 4)
    r2_south_v = _fill_to(r2_by['south'],   r1_south_v, 4)
    r2_east_v  = _fill_to(r2_by['east'],    r1_east_v,  4)
    r2_mw_v    = _fill_to(r2_by['midwest'], r1_mw_v,    4)
    r2_west_v  = _fill_to(r2_by['west'],    r1_west_v,  4)
    r2_picks = r2_south_v + r2_east_v + r2_mw_v + r2_west_v

    # ---- S16 greedy: 2 per region from R2 pool ----
    s16_by = _greedy(inner, {'south': r2_south_v, 'east': r2_east_v,
                              'midwest': r2_mw_v, 'west': r2_west_v}, 2)
    s16_south_v = _fill_to(s16_by['south'],   r2_south_v, 2)
    s16_east_v  = _fill_to(s16_by['east'],    r2_east_v,  2)
    s16_mw_v    = _fill_to(s16_by['midwest'], r2_mw_v,    2)
    s16_west_v  = _fill_to(s16_by['west'],    r2_west_v,  2)
    s16_picks = s16_south_v + s16_east_v + s16_mw_v + s16_west_v

    # ---- E8 greedy: 1 per region from S16 pool ----
    e8_by = _greedy(inner, {'south': s16_south_v, 'east': s16_east_v,
                             'midwest': s16_mw_v, 'west': s16_west_v}, 1)
    e8_south_v = _fill_to(e8_by['south'],   s16_south_v, 1)
    e8_east_v  = _fill_to(e8_by['east'],    s16_east_v,  1)
    e8_mw_v    = _fill_to(e8_by['midwest'], s16_mw_v,    1)
    e8_west_v  = _fill_to(e8_by['west'],    s16_west_v,  1)

    # JSON e8 order: [South, East, Midwest, West]
    e8_picks = [e8_south_v[0], e8_east_v[0], e8_mw_v[0], e8_west_v[0]]

    # ---- Champion and semi from labeled lines ----
    champion = _find_champion(lines, lookup)
    ff_picks  = _find_ff_picks(lines, lookup)

    e8_set = set(e8_picks)
    semi_v: list[str] = [t for t in ff_picks if t in e8_set][:2]
    for t in e8_picks:
        if len(semi_v) >= 2:
            break
        if t not in semi_v:
            semi_v.append(t)
    semi_v = semi_v[:2]

    semi_set = set(semi_v)
    if champion and champion in semi_set:
        champ = champion
    elif champion and champion in e8_set:
        champ = champion
        if champion not in semi_v:
            semi_v = [semi_v[0] if semi_v else champion, champion]
    else:
        champ = semi_v[0] if semi_v else (e8_picks[0] if e8_picks else '')

    return {
        'r1':       r1_picks,
        'r2':       r2_picks,
        's16':      s16_picks,
        'e8':       e8_picks,
        'semi':     semi_v,
        'champion': champ,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description='Convert a bracket PDF .txt file to the Brackets JSON format.',
    )
    parser.add_argument(
        '--txt',
        required=True,
        help='Path to the .txt file (e.g. Brackets/Text/MyBracket.txt)',
    )
    parser.add_argument('--name',  required=True, help='Bracket display name')
    parser.add_argument('--group', required=True, help='Group name (folder)')
    parser.add_argument(
        '--year',
        type=int,
        default=2026,
        help='Tournament year (default: 2026)',
    )
    parser.add_argument(
        '--out',
        default=None,
        help='Output .json path.  Defaults to Brackets/<group>/<safe_name>.json',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print JSON to stdout without writing a file.',
    )
    args = parser.parse_args()

    txt_path = Path(args.txt)
    if not txt_path.is_absolute():
        txt_path = repo_root / txt_path
    if not txt_path.exists():
        print(f'ERROR: txt file not found: {txt_path}', file=sys.stderr)
        sys.exit(1)

    # Load FF pairings from saved data
    ff_path = repo_root / 'Data' / 'BracketData' / str(args.year) / f'FFPairings_{args.year}.json'
    ff_pairings = '0-1,2-3'
    if ff_path.exists():
        try:
            ff_pairings = json.loads(ff_path.read_text()).get('pairings', ff_pairings)
        except Exception:
            pass

    payload = txt_to_bracket_json(
        txt_path=txt_path,
        name=args.name,
        group=args.group,
        year=args.year,
        ff_pairings=ff_pairings,
        data_root=repo_root,
    )

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    # Determine output path
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
    else:
        import re as _re
        safe_name  = _re.sub(r'[^\w\-\. ]', '', args.name).strip().replace(' ', '_')
        safe_group = _re.sub(r'[^\w\-\. ]', '', args.group).strip().replace(' ', '_')
        out_dir = repo_root / 'Brackets' / safe_group
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{safe_name}.json'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
