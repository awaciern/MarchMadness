# MarchMadness (brief)

This repository contains code and data used to train simple ML models to predict NCAA tournament games and to score historical brackets.

Status
- Main script: Python/predict_score_all_years2.py
- Data expected under: `Data/` (see "Data layout")
- Predictions written to: `Predictions2/<MODEL>/`

Quick overview
- The script loads combined historical game data, trains a scikit-learn model (selectable via CLI), then uses that trained model to predict bracket outcomes for each year and round.
- The script expects per-year KenPom data and per-round bracket CSVs to be present.

Data layout (expected relative to repo root)
- `Data/GameCombinedData/All.csv` — combined games used for training
- `Data/KenPomData/<YEAR>.csv` — KenPom/team stats for each year
- `Data/BracketCombinedData/<YEAR>/Round<ROUND>_<YEAR>.csv` — bracket pairings per round
- Outputs: `Predictions2/<MODEL>/<YEAR>.csv` and `Predictions2/<MODEL>/summary.txt`

Dependencies
1. Create and activate a virtual environment:
   - python3 -m venv venv
   - source venv/bin/activate
2. Install dependencies:
   - pip install pandas scikit-learn

Running the script
- From the repo root (recommended):
  - python Python/predict_score_all_years2.py --model logistic_lbfgs
- The script loops years 2012..2025 (skips 2020). It will:
  - Train the chosen model,
  - Predict each round for each year,
  - Write per-year CSVs under `Predictions2/<MODEL>/` and a `summary.txt`.

Supported model keys (pass to --model)
- logistic_lbfgs
- logistic_newton
- logistic_liblinear
- knn5
- svc_rbf
- svc_linear
- svc_poly2
- svc_poly3
- decision_tree
- random_forest
- adaboost
- gp

Notes
- Relative paths: adjust `--data-root` and `--output-root` CLI args if you run from a different working directory.
- If training or predictions fail, verify CSV column names and that required files are present in the `Data/` tree.
