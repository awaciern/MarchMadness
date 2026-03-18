#!/usr/bin/env python3
"""
4-model and 5-model ensemble experiments around the best combo HGB+LDA_d2+SVC_core.

Best single: HGB+LDA_d2+SVC_core = 75.87%

4-model uses: n=4, majority needs 3/4 (no ties with this logic)
5-model uses: n=5, majority needs 3/5

Note: 4-model tie resolution in ensemble3_loyo.py:
  (votes.sum * 2 > n) → for n=4, need sum >= 3, ties (2-2) go to 0
  This means tied games default to "away team wins" (or second row team).
  Use 5-model instead (n=5, always resolves with majority 3+).
"""
import subprocess, re, sys
from pathlib import Path

PYTHON = sys.executable
SCRIPT = str(Path(__file__).resolve().parent / 'ensemble3_loyo.py')

PKLS = {
    "LDA_d2":    "Predictions/13g_lda_d2full_mixup2_pca10/model.pkl",
    "LR_core":   "PredictionsModelTourney5to7_Top/8i_lr_core_mixup2_pca20_c08/model.pkl",
    "SVC_d2":    "Predictions/13g_svc_d2full_C015_mixup2_pca8/model.pkl",
    "SVC_core":  "PredictionsModelTourney5to7_Top/11b_svc_core_C0.2_mixup2_pca20/model.pkl",
    "LDA_core":  "PredictionsModelTourney5to7_Top/8c_lda_core_mixup2_pca20/model.pkl",
    "HGB":       "PredictionsModelTourney5to7_Top/9c_hgb_lr001_pca20/model.pkl",
    "ET_core":   "PredictionsModelTourney5to7_Top/9b_et_core_leaf10_mixup2_pca20/model.pkl",
}

# 3-model additional (no soft vote)
COMBOS_3 = [
    ("HGB+SVC_d2+SVC_core",    "HGB",  "SVC_d2",   "SVC_core",  None, None),
    ("HGB+ET_core+LDA_d2",     "HGB",  "ET_core",  "LDA_d2",    None, None),
    ("HGB+ET_core+SVC_core",   "HGB",  "ET_core",  "SVC_core",  None, None),
]

# 5-model (odd, no ties)
COMBOS_5 = [
    # Best 3-model + 2 more
    ("5m_HGB+LDA_d2+SVC_core+LR+LDA_c",    "HGB",  "LDA_d2",  "SVC_core", "LR_core",  "LDA_core"),
    ("5m_HGB+LDA_d2+SVC_core+LR+SVC_d2",   "HGB",  "LDA_d2",  "SVC_core", "LR_core",  "SVC_d2"),
    ("5m_HGB+LDA_d2+SVC_core+LDA_c+SVCd2", "HGB",  "LDA_d2",  "SVC_core", "LDA_core", "SVC_d2"),
]

YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

def parse_output(out, n_models):
    per_year = {}
    avg = None
    n_cols = n_models  # model columns
    for line in out.splitlines():
        # match: "  YYYY  m1  m2  m3  [m4  m5]   ensemble"
        pattern = r"\s*(\d{4})\s+" + r"[\d.]+\s+" * n_cols + r"([\d.]+)"
        m = re.match(pattern, line)
        if m:
            per_year[int(m.group(1))] = float(m.group(2))
        avg_m = re.match(r"\s*AVG\s+[\d. ]+\s+([\d.]+)$", line)
        if avg_m:
            avg = float(avg_m.group(1))
    return per_year, avg

def run_combo(p1, p2, p3, p4=None, p5=None):
    cmd = [PYTHON, SCRIPT,
           "--pkl1", PKLS[p1],
           "--pkl2", PKLS[p2],
           "--pkl3", PKLS[p3],
           "--strategy", "hard"]
    n = 3
    if p4:
        cmd += ["--pkl4", PKLS[p4]]; n += 1
    if p5:
        cmd += ["--pkl5", PKLS[p5]]; n += 1
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return parse_output(result.stdout, n)

best_avg = 0.7587  # HGB+LDA_d2+SVC_core hard vote
print(f"\n{'Combo':<47} {'AVG':>6}  {'2015':>6} {'2016':>6} {'2017':>6} {'2018':>6} {'2019':>6} {'2021':>6} {'2022':>6} {'2023':>6} {'2024':>6} {'2025':>6}")
print("-" * 145)

print("\n-- 3-model combos --")
for name, p1, p2, p3, p4, p5 in COMBOS_3:
    print(f"  Running: {name} ...", flush=True)
    try:
        per_year, avg = run_combo(p1, p2, p3, p4, p5)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {name}")
        continue
    if avg is None and per_year:
        avg = sum(per_year.values()) / len(per_year)
    yr_str = "  ".join(f"{per_year.get(y, 0):.4f}" for y in YEARS)
    flag = " ***NEW BEST***" if avg and avg > best_avg else ""
    print(f"  {name:<45} {avg:.4f}  {yr_str}{flag}")
    if avg and avg > best_avg:
        best_avg = avg

print("\n-- 5-model combos --")
for name, p1, p2, p3, p4, p5 in COMBOS_5:
    print(f"  Running: {name} ...", flush=True)
    try:
        per_year, avg = run_combo(p1, p2, p3, p4, p5)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {name}")
        continue
    if avg is None and per_year:
        avg = sum(per_year.values()) / len(per_year)
    yr_str = "  ".join(f"{per_year.get(y, 0):.4f}" for y in YEARS)
    flag = " ***NEW BEST***" if avg and avg > best_avg else ""
    print(f"  {name:<45} {avg:.4f}  {yr_str}{flag}")
    if avg and avg > best_avg:
        best_avg = avg

print(f"\nBest seen: {best_avg:.4f}")
print("=== Done ===")
