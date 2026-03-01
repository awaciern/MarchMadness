convert# MarchMadness

Trains scikit-learn models on KenPom efficiency stats to predict NCAA tournament brackets, scores historical predictions, and fills out brackets for the current year before the tournament begins.

---

## Data Pipeline

Scripts must be run in this order when setting up or updating a year:

```
1. brackets_api.py            — fetch bracket matchups/results from NCAA API
2. convert_kenpom_txt.py      — convert KenPom HTML file to KenPomData CSV
3. find_name_mismatch.py      — diagnose team name mismatches across CSVs
4. determine_winner.py        — fix WinningTeam / Team1_Win columns from scores
5. reconcile_name_mismatch.py — normalize team names so all CSVs agree
6. compile_combined_data.py   — join game/bracket data with KenPom stats
7. predict_brackets.py        — train model, simulate brackets, score results
```

---

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## Data Layout

```
Data/
  BracketData/<YEAR>/Round<N>_<YEAR>.csv   — raw bracket matchups (scraped / entered manually)
  GameData/<YEAR>.csv                       — raw per-game results
  KenPomData/<YEAR>.csv                     — KenPom efficiency stats (one row per team)
  BracketCombinedData/<YEAR>/Round<N>_<YEAR>.csv  — bracket + KenPom merged (generated)
  GameCombinedData/<YEAR>.csv               — game + KenPom merged (generated)
  GameCombinedData/All.csv                  — all years concatenated (generated)
  TeamNames/team_names.csv                  — canonical team names and aliases
```

`GameCombinedData/All.csv` is the training set.  
`BracketCombinedData` is used at simulation time to look up round matchups and results.

---

## Scripts

### `Python/determine_winner.py`

Recomputes `WinningTeam` and `Team1_Win` from `Team1_Score`/`Team2_Score` in `BracketData`. Run this if bracket CSVs have incorrect winner columns (common in pre-2015 scraped data).

```bash
python3 Python/determine_winner.py            # all years
python3 Python/determine_winner.py --year 2026
```

---

### `Python/reconcile_name_mismatch.py`

Normalizes team names across `KenPomData`, `BracketData`, and `GameData` CSVs using the alias table in `Data/TeamNames/team_names.csv`. Run after adding a new year's raw data.

```bash
python3 Python/reconcile_name_mismatch.py            # all years
python3 Python/reconcile_name_mismatch.py --year 2026
```

**Adding a new alias:** Edit `Data/TeamNames/team_names.csv`. Each row has:
```
Canonical,Alias1,Alias2,Alias3
Portland St.,Portland State,,
```
The canonical name is what KenPom uses. Aliases are replaced with the canonical in all CSVs.

---

### `Python/compile_combined_data.py`

Merges `BracketData` + `KenPomData` (into `BracketCombinedData`) and `GameData` + `KenPomData` (into `GameCombinedData`). Uses `pd.merge` to prevent silent row drops.

```bash
python3 Python/compile_combined_data.py                    # all years, both types
python3 Python/compile_combined_data.py --year 2025        # single year
python3 Python/compile_combined_data.py --type bracket     # bracket only
python3 Python/compile_combined_data.py --this-year 2026   # 2026: Round 1 bracket only, no GameData
```

`--this-year` is used before the tournament starts: it skips `GameData` (no games played yet) and only compiles `Round1` of `BracketCombinedData`.

---

### `Python/predict_brackets.py`

Trains the selected model on `GameCombinedData/All.csv`, then simulates the full bracket for each year (Rounds 1–6), scores results against actual outcomes, and writes per-year prediction CSVs.

```bash
# Score all historical years (2012–2025, excluding 2020)
python3 Python/predict_brackets.py

# Predict the current year's bracket without scoring it
python3 Python/predict_brackets.py --this-year 2026

# Choose a different model
python3 Python/predict_brackets.py --model random_forest --this-year 2026

# Override Final Four pairings for the current year (default: 0-1,2-3)
python3 Python/predict_brackets.py --this-year 2026 --final-four-pairings "0-2,1-3"
```

`--final-four-pairings` controls how the 4 predicted Elite Eight winners (indexed 0–3 in CSV order) are matched in the Final Four. Past years derive pairings automatically from the actual Round 5 data.

**Output:** `Predictions/<MODEL>/<YEAR>.csv` and `Predictions/<MODEL>/summary.txt`

#### Supported models (`--model`)

| Key | Algorithm |
|---|---|
| `logistic_lbfgs` *(default)* | Logistic Regression (LBFGS) |
| `logistic_newton` | Logistic Regression (Newton-CG) |
| `logistic_liblinear` | Logistic Regression (Liblinear) |
| `knn3` | k-Nearest Neighbors (k=3) |
| `knn5` | k-Nearest Neighbors (k=5) |
| `svc_rbf` | SVM (RBF kernel) |
| `svc_linear` | SVM (linear kernel) |
| `svc_poly2` | SVM (polynomial degree 2) |
| `svc_poly3` | SVM (polynomial degree 3) |
| `decision_tree` | Decision Tree |
| `random_forest` | Random Forest |
| `adaboost` | AdaBoost |
| `gp` | Gaussian Process |

---

### `Python/convert_kenpom_txt.py`

Converts a raw KenPom HTML file into the `KenPomData/<YEAR>.csv` format.

```bash
python3 Python/convert_kenpom_txt.py Data/KenPomRaw/3_17_2025.htm Data/KenPomData/2025.csv
# Pre-tournament (no seeds assigned yet):
python3 Python/convert_kenpom_txt.py Data/KenPomRaw/2_28_2026.htm Data/KenPomData/2026.csv --no-seeds
```

`--no-seeds` includes all 360+ teams (not just tournament field) and leaves the `Seed` column blank.

---

### `Python/brackets_api.py`

Fetches NCAA tournament bracket data from the NCAA API and writes per-round `BracketData` CSVs for a given year.

```bash
python3 Python/brackets_api.py 2025
```

---

### `Python/find_name_mismatch.py`

Diagnostic script. Compares team names across `BracketData`, `GameData`, and `KenPomData` for a given year and prints names that don't match.

```bash
python3 Python/find_name_mismatch.py --year 2026
```

---

## Adding a New Year (Pre-Tournament)

1. Copy KenPom Data → run `convert_kenpom_txt.py --no-seeds`
2. Manually enter `BracketData/<YEAR>/Round1_<YEAR>.csv` (32 matchups, scores = 0)
3. Run `reconcile_name_mismatch.py --year <YEAR>` — fix any reported mismatches in `team_names.csv`
4. Run `compile_combined_data.py --this-year <YEAR>`
5. Run `predict_brackets.py --this-year <YEAR>`

## Adding a New Year (Post-Tournament)

1. Fetch bracket results: `brackets_api.py <YEAR>` (or enter manually)
2. Run `determine_winner.py --year <YEAR>`
3. Run `reconcile_name_mismatch.py --year <YEAR>`
4. Run `compile_combined_data.py --year <YEAR>`
5. Run `predict_brackets.py` (no `--this-year`; all years scored)
