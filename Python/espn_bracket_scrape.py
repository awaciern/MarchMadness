"""
espn_bracket_scrape.py
~~~~~~~~~~~~~~~~~~~~~~
Fetch a public ESPN Tournament Challenge bracket via the Gambit API and
convert it to our bracket JSON schema.

The ESPN TC Gambit API is publicly accessible (no auth required) for any
bracket whose link was made public.

URL format:
  https://fantasy.espn.com/games/tournament-challenge-bracket-<YEAR>/bracket?id=<BRACKET_ID>

Usage (CLI):
  python3 Python/espn_bracket_scrape.py \\
      --url "https://fantasy.espn.com/games/tournament-challenge-bracket-2026/bracket?id=<id>" \\
      [--year 2026] [--out output.json] [--dry-run]

As a module:
  from espn_bracket_scrape import espn_url_to_bracket_json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
GAMBIT_BASE = "https://gambit-api.fantasy.espn.com/apis/v1"

# ESPN TC region slot ordering by displayOrder quartile:
#   displayOrder 0- 7 → EAST
#   displayOrder 8-15 → SOUTH
#   displayOrder16-23 → WEST
#   displayOrder24-31 → MIDWEST
#
# Our JSON region ordering: SOUTH=0, EAST=1, MIDWEST=2, WEST=3
#
# Mapping ESPN region index → our region index:
#   ESPN 0(EAST)    → our 1(EAST)
#   ESPN 1(SOUTH)   → our 0(SOUTH)
#   ESPN 2(WEST)    → our 3(WEST)
#   ESPN 3(MIDWEST) → our 2(MIDWEST)
_ESPN_TO_OUR_REGION = [1, 0, 3, 2]

# CSV region → start index in the 32-row Round1 CSV
# Our CSV rows: 0-7=SOUTH, 8-15=EAST, 16-23=MIDWEST, 24-31=WEST
_ESPN_REGION_TO_CSV_START = [8, 0, 24, 16]  # EAST→8, SOUTH→0, WEST→24, MIDWEST→16

# ESPN TC Final Four pairing is always:
#   FF game 1 (ESPN Matchup 1): EAST winner vs SOUTH winner
#   FF game 2 (ESPN Matchup 2): WEST winner vs MIDWEST winner
# Our encoding (SOUTH=0, EAST=1, MIDWEST=2, WEST=3):
#   FF game 1: e8[0](South) vs e8[1](East)  → pairings "0-1"
#   FF game 2: e8[2](Midwest) vs e8[3](West) → pairings "2-3"
ESPN_FF_PAIRINGS = "0-1,2-3"


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

def _parse_espn_url(url: str) -> tuple[str, str, int]:
    """
    Parse an ESPN TC bracket URL.

    Returns (bracket_id, challenge_slug, year).
    Example URL:
      https://fantasy.espn.com/games/tournament-challenge-bracket-2026/bracket?id=abc123
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    bracket_id = qs.get("id", [None])[0]
    if not bracket_id:
        raise ValueError(f"No 'id' query parameter found in URL: {url}")

    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "games":
        challenge_slug = parts[1]
    else:
        raise ValueError(f"Cannot parse challenge slug from URL: {url}")

    m = re.search(r"(\d{4})", challenge_slug)
    year = int(m.group(1)) if m else datetime.now().year

    return bracket_id, challenge_slug, year


# ---------------------------------------------------------------------------
# Gambit API helpers
# ---------------------------------------------------------------------------

def _gambit_get(path: str, timeout: int = 20) -> dict | list:
    url = f"{GAMBIT_BASE}/{path.lstrip('/')}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _get_challenge_id(challenge_slug: str) -> int:
    data = _gambit_get(f"/challenges/{challenge_slug}/?platform=chui&view=chui_default")
    try:
        return data["id"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Could not find challenge id in response: {exc}") from exc


def _get_propositions(challenge_id: int) -> list:
    return _gambit_get(
        f"/propositions/?challengeId={challenge_id}&platform=chui&view=chui_default"
    )


def _get_entry(challenge_id: int, bracket_id: str) -> dict:
    return _gambit_get(
        f"/challenges/{challenge_id}/entries/{bracket_id}/?platform=chui&view=chui_default"
    )


# ---------------------------------------------------------------------------
# Round1 CSV
# ---------------------------------------------------------------------------

def load_round1_csv(csv_path: str | Path) -> list[dict]:
    """
    Load Round1 CSV. Returns a list of 32 row dicts.
    Row order: 0-7=SOUTH, 8-15=EAST, 16-23=MIDWEST, 24-31=WEST.
    """
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "Team1": row["Team1"],
                    "Seed1": int(row.get("Team1_Seed") or 0),
                    "Team2": row["Team2"],
                    "Seed2": int(row.get("Team2_Seed") or 0),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Outcome → canonical team name mapping
# ---------------------------------------------------------------------------

def _build_outcome_map(props: list, r1_csv: list[dict]) -> dict[str, str]:
    """
    Build a full mapping: ESPN outcome_id → canonical team name.

    Strategy:
      1. From Round-1 (scoringPeriodId=1) propositions, use seed × ESPN-region
         to match each outcome to the canonical name from our Round1 CSV row.
         Also record COMPETITOR_ID → canonical for later-round cross-reference.
      2. For outcomes in all other propositions, look up by COMPETITOR_ID
         (the ESPN team identifier that is stable across all round props).
      3. Any remaining outcomes fall back to the ESPN description string.
    """
    outcome_map: dict[str, str] = {}
    competitor_to_canon: dict[str, str] = {}   # espn COMPETITOR_ID → canonical

    # --- Pass 1: seed-matched R64 outcomes ---
    r64_props = sorted(
        [p for p in props if p.get("scoringPeriodId") == 1],
        key=lambda p: p.get("displayOrder", 99),
    )

    for i, prop in enumerate(r64_props):
        espn_region = i // 8         # 0=EAST, 1=SOUTH, 2=WEST, 3=MIDWEST
        pos = i % 8
        csv_start = _ESPN_REGION_TO_CSV_START[espn_region]
        csv_row = r1_csv[csv_start + pos]

        for oc in prop.get("possibleOutcomes", []):
            seed = next(
                (int(m["value"]) for m in oc.get("mappings", []) if m["type"] == "SEED"),
                None,
            )
            comp_id = next(
                (m["value"] for m in oc.get("mappings", []) if m["type"] == "COMPETITOR_ID"),
                None,
            )
            if seed == csv_row["Seed1"]:
                canonical = csv_row["Team1"]
            elif seed == csv_row["Seed2"]:
                canonical = csv_row["Team2"]
            else:
                canonical = oc.get("description", oc.get("abbrev", ""))

            outcome_map[oc["id"]] = canonical
            if comp_id:
                competitor_to_canon[comp_id] = canonical

    # --- Pass 2: map remaining outcomes via COMPETITOR_ID ---
    for prop in props:
        for oc in prop.get("possibleOutcomes", []):
            if oc["id"] in outcome_map:
                continue
            comp_id = next(
                (m["value"] for m in oc.get("mappings", []) if m["type"] == "COMPETITOR_ID"),
                None,
            )
            if comp_id and comp_id in competitor_to_canon:
                outcome_map[oc["id"]] = competitor_to_canon[comp_id]
            else:
                outcome_map[oc["id"]] = oc.get("description", oc.get("abbrev", ""))

    return outcome_map


# ---------------------------------------------------------------------------
# Pick extraction
# ---------------------------------------------------------------------------

def _extract_picks(
    props: list, entry: dict, outcome_map: dict[str, str]
) -> dict:
    """
    Extract bracket picks from the entry and return them as a picks dict.

    Returns:
        {r1: [32], r2: [16], s16: [8], e8: [4], semi: [2], champion: str}
    """
    prop_lookup = {p["id"]: p for p in props}

    # Combine entry picks + finalPick, deduplicate by propositionId (last wins)
    all_raw = list(entry.get("picks", []))
    fp = entry.get("finalPick")
    if fp:
        all_raw.append(fp)

    seen: dict[str, dict] = {}
    for pick in all_raw:
        pid = pick.get("propositionId")
        if pid:
            seen[pid] = pick

    # Group by the PROPOSITION'S scoringPeriodId (not the pick's periodReached)
    by_period: dict[int, list] = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    for pick in seen.values():
        prop = prop_lookup.get(pick.get("propositionId"), {})
        period = prop.get("scoringPeriodId", 0)
        if period not in by_period:
            continue
        ocs = pick.get("outcomesPicked", [])
        oc_id = ocs[0].get("outcomeId") if ocs else None
        team = outcome_map.get(oc_id, "") if oc_id else ""
        by_period[period].append(
            {
                "team": team,
                "display_order": prop.get("displayOrder", 99),
                "matchup_id": prop.get("scoringPeriodMatchupId", 99),
            }
        )

    def _sorted(period: int, n: int) -> list:
        picks = by_period.get(period, [])
        key = "display_order" if period == 1 else "matchup_id"
        return sorted(picks, key=lambda x: x[key])[:n]

    def _reindex(items: list, n_per_region: int) -> list[str]:
        """
        Convert ESPN region ordering (EAST, SOUTH, WEST, MIDWEST) to our
        ordering (SOUTH, EAST, MIDWEST, WEST).
        """
        result: list[str] = [""] * len(items)
        for i, item in enumerate(items):
            espn_reg = i // n_per_region
            pos = i % n_per_region
            our_reg = _ESPN_TO_OUR_REGION[espn_reg]
            result[our_reg * n_per_region + pos] = item["team"]
        return result

    r64 = _sorted(1, 32)
    r32 = _sorted(2, 16)
    s16_raw = _sorted(3, 8)
    e8_raw = _sorted(4, 4)
    ff_raw = _sorted(5, 2)
    champ_raw = _sorted(6, 1)

    # ESPN FF Matchup 1 = EAST/SOUTH side → our semi[0] (ff_pairings 0-1)
    # ESPN FF Matchup 2 = WEST/MIDWEST side → our semi[1] (ff_pairings 2-3)
    semi = [
        ff_raw[0]["team"] if ff_raw else "",
        ff_raw[1]["team"] if len(ff_raw) > 1 else "",
    ]

    return {
        "r1":      _reindex(r64, 8),
        "r2":      _reindex(r32, 4),
        "s16":     _reindex(s16_raw, 2),
        "e8":      _reindex(e8_raw, 1),
        "semi":    semi,
        "champion": champ_raw[0]["team"] if champ_raw else "",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def espn_url_to_bracket_json(
    url: str,
    name: str = "",
    group: str = "",
    year: Optional[int] = None,
    csv_path: Optional[str | Path] = None,
) -> dict:
    """
    Fetch a public ESPN Tournament Challenge bracket and return a bracket
    payload dict compatible with our JSON schema.

    Args:
        url:      Full ESPN TC bracket URL (must contain ?id=<bracket_guid>).
        name:     Override bracket name (defaults to ESPN bracket name).
        group:    Group name for saving.
        year:     Tournament year (auto-detected from URL if omitted).
        csv_path: Path to Round1_<YEAR>.csv (auto-located if omitted).

    Returns:
        dict with keys: name, group, year, created, ff_pairings, picks
    """
    bracket_id, challenge_slug, url_year = _parse_espn_url(url)
    resolved_year = year or url_year

    if csv_path is None:
        csv_path = (
            REPO_ROOT
            / "Data"
            / "BracketData"
            / str(resolved_year)
            / f"Round1_{resolved_year}.csv"
        )

    r1_csv = load_round1_csv(csv_path)

    challenge_id = _get_challenge_id(challenge_slug)
    props = _get_propositions(challenge_id)
    entry = _get_entry(challenge_id, bracket_id)

    outcome_map = _build_outcome_map(props, r1_csv)
    picks = _extract_picks(props, entry, outcome_map)

    bracket_name = name or entry.get("name", "")

    return {
        "name":        bracket_name,
        "group":       group,
        "year":        resolved_year,
        "created":     datetime.now(timezone.utc).isoformat(),
        "ff_pairings": ESPN_FF_PAIRINGS,
        "espn_url":    url,
        "picks":       picks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a public ESPN Tournament Challenge bracket."
    )
    parser.add_argument("--url", required=True, help="ESPN TC bracket URL")
    parser.add_argument("--name", default="", help="Bracket name override")
    parser.add_argument("--group", default="", help="Group name")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output JSON file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print result without saving"
    )
    args = parser.parse_args()

    payload = espn_url_to_bracket_json(
        url=args.url,
        name=args.name,
        group=args.group,
        year=args.year,
    )

    if args.dry_run or not args.out:
        print(json.dumps(payload, indent=2))
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
