"""
rescrape_espn_brackets.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Re-fetch all bracket JSONs that have an 'espn_url' field using the current
version of espn_bracket_scrape.py.

Usage:
  python3 Python/rescrape_espn_brackets.py [--dry-run] [--dir Brackets/Me]

Options:
  --dry-run   Print what would be updated without writing files.
  --dir DIR   Only process brackets under this directory (relative to repo root).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from espn_bracket_scrape import espn_url_to_bracket_json  # noqa: E402


def rescrape(brackets_dir: Path, dry_run: bool) -> None:
    files = sorted(brackets_dir.rglob("*.json"))
    if not files:
        print("No JSON files found.")
        return

    updated = skipped = errors = 0
    for fpath in files:
        if "group_results" in fpath.name:
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP (bad JSON): {fpath.relative_to(REPO_ROOT)} — {e}")
            skipped += 1
            continue

        url = data.get("espn_url", "").strip()
        if not url:
            print(f"  SKIP (no espn_url): {fpath.relative_to(REPO_ROOT)}")
            skipped += 1
            continue

        print(f"  Fetching: {fpath.relative_to(REPO_ROOT)} ...", end=" ", flush=True)
        try:
            payload = espn_url_to_bracket_json(
                url=url,
                name=data.get("name", ""),
                group=data.get("group", ""),
                year=data.get("year"),
            )
        except Exception as e:
            print(f"ERROR — {e}")
            errors += 1
            continue

        # Preserve the original name/group (don't overwrite with ESPN's)
        payload["name"] = data.get("name", payload["name"])
        payload["group"] = data.get("group", payload["group"])
        payload["espn_url"] = url

        if dry_run:
            print("OK (dry-run, not written)")
        else:
            fpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print("OK")
        updated += 1

    print(f"\nDone — updated: {updated}, skipped: {skipped}, errors: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-scrape ESPN bracket JSONs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dir",
        default="Brackets",
        help="Directory to search (relative to repo root, default: Brackets)",
    )
    args = parser.parse_args()

    brackets_dir = REPO_ROOT / args.dir
    if not brackets_dir.is_dir():
        sys.exit(f"Directory not found: {brackets_dir}")

    rescrape(brackets_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
