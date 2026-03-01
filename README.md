# MarchMadness

Trains scikit-learn models on KenPom and BartTorvik efficiency stats to predict NCAA tournament brackets, scores historical predictions, and fills out brackets for the current year before the tournament begins.

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

## Adding a New Year (Pre-Tournament)

1. Save KenPom HTML → `convert_kenpom_txt.py --no-seeds`
2. Copy BartTorvik .txt → `convert_barttorvik_txt.py --no-seeds`
3. Manually enter `BracketData/<YEAR>/Round1_<YEAR>.csv` (32 matchups, scores = 0)
4. Run `find_name_mismatch.py --year <YEAR>` — review mismatches
5. Fix mismatches in `team_names.csv`, then run `reconcile_name_mismatch.py --year <YEAR>`; repeat until clean
6. Run `compile_combined_data.py --this-year <YEAR>`
7. Run `predict_brackets.py --this-year <YEAR>`

## Adding a New Year (Post-Tournament)

1. Fetch bracket results: `brackets_api.py <YEAR>` (or enter manually)
2. Copy final BartTorvik .txt → `convert_barttorvik_txt.py --all-teams`
3. Run `determine_winner.py --year <YEAR>`
4. Run `reconcile_name_mismatch.py --year <YEAR>`
5. Run `compile_combined_data.py --year <YEAR>`
6. Run `predict_brackets.py` (no `--this-year`; all years scored)
