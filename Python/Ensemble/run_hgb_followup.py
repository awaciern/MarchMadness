#!/usr/bin/env python3
"""
Follow-up HGB combos after discovering HGB+LDA_d2+SVC_core = 75.87%

Key insight: HGB + LDA_d2 + SVC_core works because:
- LDA_d2: strong 2015/2019 (d2full features)
- SVC_core: strong 2021/2022 (full-season FEAT_CORE)
- HGB: strong 2015/2018/2025 (tree-based, different error mode)
- HGB+SVC_core vote together to override LDA_d2 mistakes in 2021/2018

Next to try:
1. Soft vote: HGB+LDA_d2+SVC_core with soft strategy (calibrated proba)
2. 5-model: HGB + LDA_d2 + SVC_core + LR_core + LDA_core
3. 4-model: HGB + LDA_d2 + SVC_core + LR_core
4. 4-model: HGB + LDA_d2 + SVC_core + SVC_d2  (adds d2full coverage)
5. HGB + LDA_d2 + SVC_core + ET_core (5 models if add ET)
6. OTHER new 3-model with HGB: try replacing LDA_d2 with SVC_d2 (same feats, diff model key)
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

# 3-model combos to add
COMBOS_3 = [
    # Soft vote on the new best combo
    ("HGB+LDA_d2+SVC_core_SOFT",   "HGB",  "LDA_d2",   "SVC_core",   None, "soft"),
    # Replace LDA_d2 with SVC_d2 (same feature space, different model) 
    ("HGB+SVC_d2+SVC_core",        "HGB",  "SVC_d2",   "SVC_core",   None, "hard"),
    # Add ET_core as alternative
    ("HGB+ET_core+LDA_d2",         "HGB",  "ET_core",  "LDA_d2",     None, "hard"),
    ("HGB+ET_core+SVC_core",       "HGB",  "ET_core",  "SVC_core",   None, "hard"),
]

# 4-model combos (HGB as 4th model, odd logic: may have ties)
COMBOS_4 = [
    ("HGB+LDA_d2+SVC_core+LR_core",      "HGB",  "LDA_d2",  "SVC_core", "LR_core",  "hard"),
    ("HGB+LDA_d2+SVC_core+SVC_d2",       "HGB",  "LDA_d2",  "SVC_core", "SVC_d2",   "hard"),
    ("HGB+LDA_d2+SVC_core+LDA_core",     "HGB",  "LDA_d2",  "SVC_core", "LDA_core", "hard"),
]

YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

def run_combo(p1, p2, p3, p4=None, strategy="hard"):
    cmd = [PYTHON, SCRIPT,
           "--pkl1", PKLS[p1],
           "--pkl2", PKLS[p2],
           "--pkl3", PKLS[p3],
           "--strategy", strategy]
    if p4:
        cmd += ["--pkl4", PKLS[p4]]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=360)
    out = result.stdout
    per_year = {}
    avg = None
    # 3-model: "  2015  0.8413  0.7302  0.6984     0.8413"
    # 4-model: "  2015  0.8413  0.7302  0.6984  0.7619     0.8413"
    for line in out.splitlines():
        m3 = re.match(r"\s*(\d{4})\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", line)
        m4 = re.match(r"\s*(\d{4})\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", line)
        m = m4 if p4 else m3
        if m:
            per_year[int(m.group(1))] = float(m.group(2))
        avg_re = re.match(r"\s*AVG\s+[\d. ]+\s+([\d.]+)$", line)
        if avg_re:
            avg = float(avg_re.group(1))
    return per_year, avg

best_avg = 0.7587  # HGB+LDA_d2+SVC_core
print(f"\n{'Combo':<40} {'AVG':>6}  {'2015':>6} {'2016':>6} {'2017':>6} {'2018':>6} {'2019':>6} {'2021':>6} {'2022':>6} {'2023':>6} {'2024':>6} {'2025':>6}")
print("-" * 135)

for name, p1, p2, p3, p4, strat in COMBOS_3 + COMBOS_4:
    print(f"  Running: {name} ...", flush=True)
    try:
        per_year, avg = run_combo(p1, p2, p3, p4, strat)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {name}")
        continue
    if avg is None and per_year:
        avg = sum(per_year.values()) / len(per_year)
    yr_str = "  ".join(f"{per_year.get(y, 0):.4f}" for y in YEARS)
    flag = " ***NEW BEST***" if avg and avg > best_avg else ""
    print(f"  {name:<38} {avg:.4f}  {yr_str}{flag}")
    if avg and avg > best_avg:
        best_avg = avg

print(f"\nBest seen: {best_avg:.4f}")
print("=== Done ===")
