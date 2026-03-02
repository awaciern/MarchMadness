# MarchMadness

Trains scikit-learn models on KenPom/BartTorvik efficiency stats to predict NCAA tournament brackets, scores historical predictions, and fills out brackets for the current year before the tournament begins. Includes a web UI (`app.py`) for interactive model building and result exploration.

---

## Data Pipeline

Scripts must be run in this order when setting up or updating a year:

```
1. brackets_api.py              — fetch bracket matchups/results from NCAA API
2. convert_kenpom_txt.py        — convert KenPom HTML file to KenPomData CSV
3. convert_barttorvik_txt.py    — convert BartTorvik .txt file to BartTorvikData CSV
4. find_name_mismatch.py        — diagnose team name mismatches across CSVs
5. determine_winner.py          — fix WinningTeam / Team1_Win columns from scores
6. reconcile_name_mismatch.py   — normalize team names so all CSVs agree
7. compile_combined_data.py     — join game/bracket data with KenPom + BartTorvik stats
8. predict_brackets.py          — train model, simulate brackets, score results
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
  BracketData/<YEAR>/Round<N>_<YEAR>.csv          — raw bracket matchups (scraped / entered manually)
  GameData/<YEAR>.csv                              — raw per-game results
  KenPomData/<YEAR>.csv                            — KenPom efficiency stats (one row per team)
  BartTorvikData/<YEAR>.csv                        — BartTorvik efficiency stats (one row per team)
  BracketCombinedData/<YEAR>/Round<N>_<YEAR>.csv   — bracket + KP + BT merged (generated)
  GameCombinedData/<YEAR>.csv                      — game + KP + BT merged (generated)
  GameCombinedData/All.csv                         — all years concatenated (generated)
  TeamNames/team_names.csv                         — canonical team names and aliases
```

`GameCombinedData/All.csv` is the training set.  
`BracketCombinedData` is used at simulation time to look up round matchups and stats.

### Column naming convention

Combined CSVs use source-prefixed column names so both stat sources can coexist:

- `KP__<stat>__1` / `KP__<stat>__2` — KenPom stat for Team 1 / Team 2
- `BT__<stat>__1` / `BT__<stat>__2` — BartTorvik stat for Team 1 / Team 2

Example: `KP__AdjO__1`, `BT__Barthag__2`.

---

## Scripts

### `Python/convert_kenpom_txt.py`

Converts a raw KenPom HTML file (saved from the browser) into the `KenPomData/<YEAR>.csv` format.

```bash
python3 Python/convert_kenpom_txt.py Data/KenPomRaw/3_17_2025.htm Data/KenPomData/2025.csv
# Pre-tournament (no seeds assigned yet):
python3 Python/convert_kenpom_txt.py Data/KenPomRaw/2_28_2026.htm Data/KenPomData/2026.csv --no-seeds
```

`--no-seeds` includes all 360+ teams and leaves the `Seed` column blank.

---

### `Python/convert_barttorvik_txt.py`

Converts a raw BartTorvik .txt file (copied from barttorvik.com) into the `BartTorvikData/<YEAR>.csv` format.

```bash
python3 Python/convert_barttorvik_txt.py Data/BartTorvikData/2026.txt Data/BartTorvikData/2026.csv
# Pre-tournament (no seeds, include all teams):
python3 Python/convert_barttorvik_txt.py Data/BartTorvikData/2026.txt Data/BartTorvikData/2026.csv --no-seeds
# Include non-tournament teams (post-tournament archive):
python3 Python/convert_barttorvik_txt.py Data/BartTorvikData/2025.txt Data/BartTorvikData/2025.csv --all-teams
```

| Flag | Behavior |
|---|---|
| *(default)* | Tournament-seeded teams only |
| `--no-seeds` | All teams; `Seed` left blank (use pre-tournament) |
| `--all-teams` | All teams including non-tournament; `Seed` blank for unseeded |

---

### `Python/brackets_api.py`

Fetches NCAA tournament bracket data from the NCAA API and writes per-round `BracketData` CSVs for a given year.

```bash
python3 Python/brackets_api.py 2025
```

---

### `Python/find_name_mismatch.py`

Diagnostic script. Compares team names in `BracketData/Round1` against both `KenPomData` and `BartTorvikData` for a given year and prints any names that don't match.

```bash
python3 Python/find_name_mismatch.py --year 2026
```

---

### `Python/determine_winner.py`

Recomputes `WinningTeam` and `Team1_Win` from `Team1_Score`/`Team2_Score` in `BracketData`. Run this if bracket CSVs have incorrect winner columns (common in pre-2015 scraped data).

```bash
python3 Python/determine_winner.py            # all years
python3 Python/determine_winner.py --year 2026
```

---

### `Python/reconcile_name_mismatch.py`

Normalizes team names across `KenPomData`, `BartTorvikData`, `BracketData`, and `GameData` CSVs using the alias table in `Data/TeamNames/team_names.csv`. Run after adding a new year's raw data.

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

Joins `BracketData` and `GameData` with both `KenPomData` and `BartTorvikData`, writing combined files to `BracketCombinedData` and `GameCombinedData`. BartTorvik columns use a left join so rows compile even when BartTorvik data is absent for a year.

```bash
python3 Python/compile_combined_data.py                    # all years, both types
python3 Python/compile_combined_data.py --year 2025        # single year
python3 Python/compile_combined_data.py --type bracket     # bracket only
python3 Python/compile_combined_data.py --type game        # game only
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

# Use BartTorvik as the source for common features (AdjO, AdjD, etc.)
python3 Python/predict_brackets.py --expert barttorvik --this-year 2026

# Override feature set
python3 Python/predict_brackets.py --features WinPct AdjO AdjD Barthag SOS_AdjEM

# Override Final Four pairings for the current year (default: 0-1,2-3)
python3 Python/predict_brackets.py --this-year 2026 --final-four-pairings "0-2,1-3"
```

`--final-four-pairings` controls how the 4 predicted Elite Eight winners (indexed 0–3 in CSV order) are matched in the Final Four. Past years derive pairings automatically from the actual Round 5 data.

**Output:** `Predictions/<model>_<score>_<KP|BT>_<feat1>+<feat2>+.../<YEAR>.csv` and `summary.txt`

#### `--expert`

Controls which source supplies the *common* stats (those available from both KenPom and BartTorvik: `WinPct`, `AdjO`, `AdjD`, `AdjT`, `Conf`, `Wins`, `Losses`, and their rank variants). KenPom-only or BartTorvik-only features always use their own source regardless of this flag. Default: `kenpom`.

#### `--features`

Space-separated list of unprefixed base feature names. The source prefix (`KP__` or `BT__`) is resolved automatically based on `--expert`.

| Group | Features |
|---|---|
| Common (source = `--expert`) | `WinPct` `Wins` `Losses` `AdjO` `Rk_AdjO` `AdjD` `Rk_AdjD` `AdjT` `Rk_AdjT` `Conf` |
| KenPom-only (always `KP__`) | `AdjEM` `Rk_AdjEM` `Luck` `Rk_Luck` `SOS_AdjEM` `Rk_SOS_AdjEM` `SOS_AdjO` `Rk_SOS_AdjO` `SOS_AdjD` `Rk_SOS_AdjD` `NCSOS_AdjEM` `Rk_NCSOS_AdjEM` |
| BartTorvik-only (always `BT__`) | `Barthag` `Rk_Barthag` `EFG%` `EFGD%` `TOR` `TORD` `ORB` `DRB` `FTR` `FTRD` `2P%` `2P%D` `3P%` `3P%D` `3PR` `3PRD` `WAB` `Rk_WAB` `ConfWinPct` `ConfWins` `ConfLosses` (and their `Rk_` variants) |
| Categorical opt-in | `Conf` `Seed` |

Default features: `WinPct AdjO AdjD SOS_AdjEM`

#### Supported models (`--model`)

| Key | Algorithm | sklearn docs |
|---|---|---|
| `logistic_regression` *(default)* | Logistic Regression | [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) |
| `knn` | k-Nearest Neighbors | [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) |
| `svc` | Support Vector Machine | [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) |
| `decision_tree` | Decision Tree | [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) |
| `random_forest` | Random Forest | [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) |
| `adaboost` | AdaBoost | [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) |
| `gpc` | Gaussian Process | [GaussianProcessClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html) |

#### `--model-params`

Pass constructor keyword arguments to the model using `key=value` pairs. Values are automatically cast to `int`, `float`, `bool`, `None`, or `str`. See the sklearn docs linked above for valid parameters for each model.

```bash
python3 Python/predict_brackets.py --model logistic_regression --model-params solver=lbfgs max_iter=1000 random_state=0
python3 Python/predict_brackets.py --model knn --model-params n_neighbors=7
python3 Python/predict_brackets.py --model random_forest --model-params n_estimators=200 random_state=42
python3 Python/predict_brackets.py --model svc --model-params kernel=rbf C=1.0
```

Model parameters are appended to the output folder name: `Predictions/<model>_<score>_<KP|BT>_<features>_<param1=val1>+<param2=val2>/`

---

### `Python/app.py` — Web UI

A browser-based interface for building models and exploring results without using the command line.

```bash
python3 Python/app.py
# Open http://localhost:5050
```

#### Features

**Create & run a model**

The left-hand form mirrors all `predict_brackets.py` options:

| Field | Description |
|---|---|
| Model Type | Choose from all 7 supported models |
| Model Parameters | Optional `key=value` pairs (e.g. `n_estimators=200 random_state=42`) |
| Stats Source | KenPom or BartTorvik for common features |
| Features | Click-to-toggle chip grid; color-coded by source (blue = common, purple = KenPom-only, pink = BartTorvik-only, green = metadata) |

Clicking **Run Prediction** executes `predict_brackets.py --this-year 2026` in the background. A live log streams output line-by-line in the right panel as it runs.

**View results**

Once a run completes (or when clicking any saved model):
- **Year links** — click any year to open its full HTML bracket visual in a new tab; the current year (2026) is highlighted with a star
- **Summary** — displays `summary.txt` showing per-round accuracy, avg bracket score, and train/test accuracy across all years

**Existing models table**

A sortable table at the top of the page lists every folder in `Predictions/` (sorted by bracket score, highest first). Each row shows rank, score, model type, expert source, features, and any custom parameters. Clicking a row loads that model's summary and year links instantly without re-running anything.

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
