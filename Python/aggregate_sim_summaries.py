#!/usr/bin/env python3
import csv
from pathlib import Path
import re
import shutil

# Search for any PredictionsSimTourney* folders at repo root and aggregate them.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PREFIX = 'PredictionsSimTourney'
DEST_ROOT = REPO_ROOT / 'Predictions'
DEST_ROOT.mkdir(exist_ok=True)
OUT = DEST_ROOT / 'aggregate_summary.csv'

rows = []
seen = set()
for candidate in sorted(REPO_ROOT.iterdir()):
    if not candidate.is_dir():
        continue
    if not candidate.name.startswith(SRC_PREFIX):
        continue
    for child in sorted(candidate.iterdir()):
        if not child.is_dir():
            continue
        summary = child / 'summary.txt'
        if not summary.exists():
            continue
        text = summary.read_text(encoding='utf-8')
        # find Avg LOYO train/test
        m_train = re.search(r'Avg LOYO train acc ?:?\s*([0-9.]+)', text)
        m_test = re.search(r'Avg LOYO test acc ?:?\s*([0-9.]+)', text)
        sim_train = re.search(r'Sim\+real train acc:?\s*([0-9.]+)', text)
        sim_real_test = re.search(r'Real-only test acc:?\s*([0-9.]+)', text)
        avg_bracket = re.search(r'Avg bracket score:?\s*([0-9.]+)', text)
        run_name = child.name
        if run_name in seen:
            # Skip duplicate run names (prefer first encountered)
            continue
        seen.add(run_name)
        rows.append({
            'run': run_name,
            'avg_loyo_train': float(m_train.group(1)) if m_train else '',
            'avg_loyo_test': float(m_test.group(1)) if m_test else '',
            'sim_plus_real_train': float(sim_train.group(1)) if sim_train else '',
            'sim_plus_real_real_test': float(sim_real_test.group(1)) if sim_real_test else '',
            'avg_bracket_score': float(avg_bracket.group(1)) if avg_bracket else '',
        })

        # Copy run folder into repo-level Predictions for the web UI to consume.
        dest = DEST_ROOT / run_name
        if dest.exists():
            # remove and replace to ensure latest files
            shutil.rmtree(dest)
        shutil.copytree(child, dest)

with OUT.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['run','avg_loyo_train','avg_loyo_test','sim_plus_real_train','sim_plus_real_real_test','avg_bracket_score'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f'Wrote {OUT} with {len(rows)} runs')
