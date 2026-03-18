#!/usr/bin/env python3
"""
Run remaining high-priority ensemble combos and print a summary table.
"""
import subprocess, sys, re
from pathlib import Path

PYTHON = sys.executable
SCRIPT = str(Path(__file__).resolve().parent / 'ensemble3_loyo.py')

PKLS = {
    "LDA_d2":    "Predictions/13g_lda_d2full_mixup2_pca10/model.pkl",
    "LR_core":   "PredictionsModelTourney5to7_Top/8i_lr_core_mixup2_pca20_c08/model.pkl",
    "SVC_d2":    "Predictions/13g_svc_d2full_C015_mixup2_pca8/model.pkl",
    "SVC_core":  "PredictionsModelTourney5to7_Top/11b_svc_core_C0.2_mixup2_pca20/model.pkl",
    "LDA_core":  "PredictionsModelTourney5to7_Top/8c_lda_core_mixup2_pca20/model.pkl",
    "GBM_d2":    "Predictions/15a_gbm_d2full_nest300_lr005_d3_mixup2/model.pkl",
    "GNB":       "Predictions/17d_gnb_core_pca20_1e9_mixup2/model.pkl",
}

COMBOS = [
    # New untested combos
    ("LDA_d2+LDA_core+SVC_core",  "LDA_d2",   "LDA_core",  "SVC_core",  None),
    ("LDA_core+SVC_d2+SVC_core",  "LDA_core",  "SVC_d2",   "SVC_core",  None),
    ("LDA_d2+LR_core+GBM_d2",     "LDA_d2",   "LR_core",   "GBM_d2",   None),
    ("GBM_d2+SVC_core+LR_core",   "GBM_d2",   "SVC_core",  "LR_core",  None),
    ("LDA_d2+GBM_d2+SVC_core",    "LDA_d2",   "GBM_d2",    "SVC_core",  None),
    ("LDA_core+LR_core+SVC_core",  "LDA_core", "LR_core",   "SVC_core",  None),  # all-core
    ("GNB+LDA_d2+LR_core",        "GNB",      "LDA_d2",    "LR_core",   None),
    ("GNB+LDA_core+SVC_core",     "GNB",      "LDA_core",  "SVC_core",  None),
]

YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

def run_combo(name, p1, p2, p3, p4=None):
    cmd = [PYTHON, SCRIPT,
           "--pkl1", PKLS[p1],
           "--pkl2", PKLS[p2],
           "--pkl3", PKLS[p3],
           "--strategy", "hard"]
    if p4:
        cmd += ["--pkl4", PKLS[p4]]
    result = subprocess.run(cmd, capture_output=True, text=True)
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

print(f"\n{'Combo':<35} {'AVG':>6}  {'2015':>6} {'2016':>6} {'2017':>6} {'2018':>6} {'2019':>6} {'2021':>6} {'2022':>6} {'2023':>6} {'2024':>6} {'2025':>6}")
print("-" * 130)

best_avg = 0.7524  # current best
for combo_name, p1, p2, p3, p4 in COMBOS:
    print(f"  Running: {combo_name} ...", flush=True)
    per_year, avg = run_combo(combo_name, p1, p2, p3, p4)
    if avg is None and per_year:
        avg = sum(per_year.values()) / len(per_year)
    yr_str = "  ".join(f"{per_year.get(y, 0):.4f}" for y in YEARS)
    flag = " ***NEW BEST***" if avg and avg > best_avg else ""
    print(f"  {combo_name:<33} {avg:.4f}  {yr_str}{flag}")
    if avg and avg > best_avg:
        best_avg = avg

print(f"\nBest seen: {best_avg:.4f}")
print("=== Runs complete ===")
