# MarchMadness (brief)

This repository contains code and data used to train simple ML models to predict NCAA tournament games and to score historical brackets.

Status
- Main script: Python/predict_score_all_years2.py
- Data expected under: `Data/` (see "Data layout")
- Predictions written to: `Predictions2/<MODEL>/`

Quick overview
- The script loads combined historical game data, trains a scikit-learn model (you must select/uncomment one), then uses that trained model to predict bracket outcomes for each year and round.
- The script expects per-year KenPom data and per-round bracket CSVs to be present.

Data layout (expected relative to `Python/` directory)
- `../Data/GameCombinedData/All.csv` — combined games used for training
- `../Data/KenPomData/<YEAR>.csv` — KenPom/team stats for each year
- `../Data/BracketCombinedData/<YEAR>/Round<ROUND>_<YEAR>.csv` — bracket pairings per round
- Outputs: `../Predictions2/<MODEL>/<YEAR>.csv` and `../Predictions2/<MODEL>/summary.txt`

How to prepare environment
1. Create and activate a virtual environment:
   - python3 -m venv venv
   - source venv/bin/activate
2. Install dependencies:
   - pip install pandas scikit-learn

How to select a model
- Open `Python/predict_score_all_years2.py`
- Near the top there are many example model lines commented out, for example:
  - `# MODEL = 'Logistic_LBFGS'`
  - `# model = LogisticRegression(...).fit(X_train, y_train)`
- You must uncomment a matching pair so:
  - `MODEL` is defined (string used to name the output directory), and
  - `model` is trained (the `.fit(X_train, y_train)` call must be present).
- Example (edit the file and uncomment or add these lines after the train/test split):
  ```
  MODEL = 'Logistic_LBFGS'
  model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000).fit(X_train, y_train)
  ```
- If `MODEL` or `model` are not defined, the script will raise an error.

Running the script
- From the `Python/` directory:
  - python predict_score_all_years2.py
- The script loops years 2012..2025 (skips 2020). It will:
  - Train (if you left training lines uncommented),
  - Predict each round for each year,
  - Write per-year CSV in `../Predictions2/<MODEL>/` and a `summary.txt`.

Common pitfalls
- Relative paths: run the script from the `Python/` folder (or adjust paths).
- Missing CSV files: ensure all referenced CSVs exist (KenPom and bracket data).
- Ensure target column names and feature names in the data match those referenced in the script.

Tips
- To try different models, change `MODEL` and uncomment a corresponding `model = ... .fit(...)` line.
- To debug missing columns or join mismatches, print shapes after joins (the script already has commented prints you can enable).

License / notes
- This README is minimal. Adjust as needed for additional scripts or data processing steps.
