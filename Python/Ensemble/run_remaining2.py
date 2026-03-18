#!/usr/bin/env python3
"""Run the remaining ensemble combos (excluding GBM which is slow)."""
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
    "GNB":       "Predictions/17d_gnb_core_pca20_1e9_mixup2/model.pkl",
    "ET_core":   "PredictionsModelTourney5to7_Top/9b_et_core_leaf10_mixup2_pca20/model.pkl",
    "HGB":       "PredictionsModelTourney5to7_Top/9c_hgb_lr001_pca20/model.pkl",
}

COMBOS = [
    ("GBM_d2+SVC_core+LR_core",   "GBM"),   # skip — GBM is slow
    ("LDA_d2+GBM_d2+SVC_core",    "GBM"),   # skip
    ("LDA_core+LR_core+SVC_core",  None),
    ("GNB+LDA_d2+LR_core",         None),
    ("GNB+LDA_core+SVC_core",      None),
    ("ET_core+LDA_d2+LR_core",     None),
    ("ET_core+LDA_d2+SVC_d2",      None),
    ("HGB+LDA_d2+LR_core",         None),
]

KEYS = {
    "LDA_core+LR_core+SVC_core":  ("LDA_core",  "LR_core",   "SVC_core"),
    "GNB+LDA_d2+LR_core":         ("GNB",       "LDA_d2",    "LR_core"),
    "GNB+LDA_core+SVC_core":      ("GNB",       "LDA_core",  "SVC_core"),
    "ET_core+LDA_d2+LR_core":     ("ET_core",   "LDA_d2",    "LR_core"),
    "ET_core+LDA_d2+SVC_d2":      ("ET_core",   "LDA_d2",    "SVC_d2"),
    "HGB+LDA_d2+LR_core":         ("HGB",       "LDA_d2",    "LR_core"),
}

YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

def run_combo(name, p1, p2, p3):
    cmd = [PYTHON, SCRIPT,
           "--pkl1", PKLS[p1],
           "--pkl2", PKLS[p2],
           "--pkl3", PKLS[p3],
           "--strategy", "hard"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = result.stdout
    per_year = {}
    avg = None
    for line in out.splitlines():
        m = re.match(r"\s*(\d{4})\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", line)
        if m:
            per_year[int(m.group(1))] = float(m.group(2))
        m2 = re.match(r"\s*AVG\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", line)
        if m2:
            avg = float(m2.group(1))
    return per_year, avg

best_avg = 0.7524
print(f"\n{'Combo':<35} {'AVG':>6}  {'2015':>6} {'2016':>6} {'2017':>6} {'2018':>6} {'2019':>6} {'2021':>6} {'2022':>6} {'2023':>6} {'2024':>6} {'2025':>6}")
print("-" * 130)

for name, skip in COMBOS:
    if skip == "GBM":
        print(f"  SKIPPED (GBM slow): {name}")
        continue
    p1, p2, p3 = KEYS[name]
    print(f"  Running: {name} ...", flush=True)
    try:
        per_year, avg = run_combo(name, p1, p2, p3)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {name}")
        continue
    if avg is None and per_year:
        avg = sum(per_year.values()) / len(per_year)
    yr_str = "  ".join(f"{per_year.get(y, 0):.4f}" for y in YEARS)
    flag = " ***NEW BEST***" if avg and avg > best_avg else ""
    print(f"  {name:<33} {avg:.4f}  {yr_str}{flag}")
    if avg and avg > best_avg:
        best_avg = avg

print(f"\nBest seen: {best_avg:.4f}")
print("=== Done ===")
