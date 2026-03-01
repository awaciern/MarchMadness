"""
predict_brackets.py

Trains a model (leave-one-year-out to prevent data leakage), simulates filling
out the tournament bracket from Round 1 forward, and evaluates accuracy for each
historical year.  The current year's bracket is also filled out and saved.

Final Four pairings (Round 5)
- Past years: derived automatically from the actual Round 5 CSV by comparing its
  teams against the predicted Elite Eight winners.
- Current year: controlled by --final-four-pairings (default "0-1,2-3"), which
  specifies how the 4 predicted regional winners (indexed 0-3 in Elite Eight order)
  are matched up.  E.g. "0-2,1-3" means winner[0] vs winner[2] and winner[1] vs
  winner[3].

Usage:
    python3 predict_brackets.py --model logistic_lbfgs
    python3 predict_brackets.py -m random_forest --final-four-pairings "0-2,1-3"
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_YEARS = [y for y in range(2012, 2026) if y != 2020]  # all completed years through 2025

FEATURE_LIST = [
    'WinPct__1', 'AdjEM__1', 'AdjO__1', 'AdjD__1', 'AdjT__1', 'Luck__1',
    'SOS_AdjEM__1', 'Rk_NCSOS_AdjEM__1',
    'WinPct__2', 'AdjEM__2', 'AdjO__2', 'AdjD__2', 'AdjT__2', 'Luck__2',
    'SOS_AdjEM__2', 'Rk_NCSOS_AdjEM__2',
]

MODEL_REGISTRY = {
    'logistic_lbfgs':     lambda: LogisticRegression(random_state=0, solver='lbfgs',     max_iter=1000),
    'logistic_newton':    lambda: LogisticRegression(random_state=0, solver='newton-cg', max_iter=1000),
    'logistic_liblinear': lambda: LogisticRegression(random_state=0, solver='liblinear', max_iter=1000),
    'knn3':               lambda: KNeighborsClassifier(n_neighbors=3),
    'knn5':               lambda: KNeighborsClassifier(n_neighbors=5),
    'svc_rbf':            lambda: SVC(gamma='auto'),
    'svc_linear':         lambda: SVC(kernel='linear'),
    'svc_poly2':          lambda: SVC(kernel='poly', degree=2),
    'svc_poly3':          lambda: SVC(kernel='poly', degree=3),
    'decision_tree':      lambda: DecisionTreeClassifier(),
    'random_forest':      lambda: RandomForestClassifier(),
    'adaboost':           lambda: AdaBoostClassifier(),
    'gp':                 lambda: GaussianProcessClassifier(),
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_combined_games(data_root: Path, exclude_year: int = None) -> pd.DataFrame:
    """Load GameCombinedData/All.csv, optionally excluding one year."""
    df = pd.read_csv(data_root / 'Data' / 'GameCombinedData' / 'All.csv')
    if exclude_year is not None:
        df = df[df['Year'] != exclude_year]
    return df


def load_bracket_round(data_root: Path, year: int, rnd: int) -> pd.DataFrame:
    return pd.read_csv(
        data_root / 'Data' / 'BracketCombinedData' / str(year) / f'Round{rnd}_{year}.csv'
    )


def load_kenpom(data_root: Path, year: int) -> pd.DataFrame:
    return pd.read_csv(data_root / 'Data' / 'KenPomData' / f'{year}.csv')


def attach_kenpom(df_matchups: pd.DataFrame, df_kp: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns Team__1 and Team__2, merge in all KenPom
    stats (suffixed __1 and __2).
    """
    kp1 = df_kp.add_suffix('__1')
    kp2 = df_kp.add_suffix('__2')
    df = df_matchups.merge(kp1, on='Team__1', how='inner')
    df = df.merge(kp2, on='Team__2', how='inner')
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Final Four pairing helpers
# ---------------------------------------------------------------------------

def derive_ff_pairings_from_data(data_root: Path, year: int) -> List[Tuple[int, int]]:
    """
    For a past year, read the actual Round 5 CSV and compare its teams to the
    actual Round 4 winners to figure out which indices (0-3) are paired.

    Returns a list of two (i, j) tuples, e.g. [(0, 1), (2, 3)].
    """
    df4 = load_bracket_round(data_root, year, 4)
    df5 = load_bracket_round(data_root, year, 5)

    actual_r4_winners = [
        row['Team__1'] if row['Win__1'] else row['Team__2']
        for _, row in df4.iterrows()
    ]

    pairings = []
    for _, game in df5.iterrows():
        t1, t2 = game['Team__1'], game['Team__2']
        try:
            i = actual_r4_winners.index(t1)
            j = actual_r4_winners.index(t2)
        except ValueError as e:
            raise RuntimeError(
                f'{year} Round 5 team not found in Round 4 winners: {e}\n'
                f'Round 4 winners: {actual_r4_winners}'
            )
        pairings.append((i, j))

    return pairings


def parse_ff_pairings_arg(arg: str) -> List[Tuple[int, int]]:
    """
    Parse a string like "0-1,2-3" or "0-2,1-3" into [(0,1),(2,3)].
    """
    pairings = []
    for part in arg.split(','):
        a, b = part.strip().split('-')
        pairings.append((int(a), int(b)))
    if len(pairings) != 2:
        raise ValueError(f'Expected exactly 2 pairings, got: {arg}')
    return pairings


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def build_and_train_model(model_key: str, X_train: pd.DataFrame, y_train: pd.Series):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Options: {list(MODEL_REGISTRY)}")
    estimator = MODEL_REGISTRY[model_key]()
    estimator.fit(X_train, y_train)
    return estimator


# ---------------------------------------------------------------------------
# Bracket simulation
# ---------------------------------------------------------------------------

def simulate_bracket(
    model,
    data_root: Path,
    year: int,
    this_year: int = None,
    ff_pairings: List[Tuple[int, int]] = None,
) -> Tuple[list, list, list, list, int]:
    """
    Simulate filling out a bracket from Round 1 using the model.

    Returns:
        pred_teams_by_round   – list of 6 lists of team names
        pred_seeds_by_round   – list of 6 lists of seeds
        correct_by_round      – list of 6 lists of bools (empty list for current year)
        num_correct_by_round  – list of 6 ints  (zeros for current year)
        score                 – total ESPN-style bracket score (0 for current year)
    """
    is_current = (this_year is not None and year == this_year)
    if ff_pairings is None:
        ff_pairings = [(0, 1), (2, 3)]

    pred_teams_by_round: list = []
    pred_seeds_by_round: list = []
    correct_by_round:    list = []
    num_correct_by_round: list = []
    total_score = 0

    # The predicted winners from the previous round (used to build rnd 2+ matchups).
    prev_pred_teams: List[str] = []
    # Seed lookup built from Round 1 so later rounds can fill in seeds when KenPom
    # has blank seeds (e.g. current year loaded with --no-seeds).
    team_seed_map: dict = {}

    for rnd in range(1, 7):
        if rnd == 1 or not is_current:
            # For past years we always load the actual bracket CSV (which contains
            # the real Winning_Team column) to evaluate accuracy.
            df_round = load_bracket_round(data_root, year, rnd)
        # For rounds 2-6 of the current year, df_round is built from predictions.

        if rnd == 1:
            # Populate seed map from Round 1 for use in later rounds.
            for _, _row in df_round.iterrows():
                team_seed_map[_row['Team__1']] = _row['Seed__1']
                team_seed_map[_row['Team__2']] = _row['Seed__2']

        if rnd > 1:
            # Capture actual winning teams before rebuilding df_round (past years only).
            actual_winners = (
                df_round['Winning_Team'].reset_index(drop=True)
                if not is_current else None
            )

            # Determine matchup order for this round.
            if rnd == 5:
                ordered = [None] * 4
                for (i, j) in ff_pairings:
                    # Place team at position i as Team__1 and j as Team__2.
                    ordered[i] = prev_pred_teams[i]
                    ordered[j] = prev_pred_teams[j]
                matchup_teams = [
                    (prev_pred_teams[ff_pairings[0][0]], prev_pred_teams[ff_pairings[0][1]]),
                    (prev_pred_teams[ff_pairings[1][0]], prev_pred_teams[ff_pairings[1][1]]),
                ]
            else:
                matchup_teams = [
                    (prev_pred_teams[i], prev_pred_teams[i + 1])
                    for i in range(0, len(prev_pred_teams), 2)
                ]

            df_matchups = pd.DataFrame(matchup_teams, columns=['Team__1', 'Team__2'])
            df_kp = load_kenpom(data_root, year)
            df_round = attach_kenpom(df_matchups, df_kp)

            # Fill seeds from map in case KenPom has blank seeds (e.g. --no-seeds year).
            if df_round['Seed__1'].isna().any() or df_round['Seed__2'].isna().any():
                df_round['Seed__1'] = df_round['Team__1'].map(team_seed_map).fillna(df_round['Seed__1'])
                df_round['Seed__2'] = df_round['Team__2'].map(team_seed_map).fillna(df_round['Seed__2'])

            if not is_current:
                df_round['Winning_Team'] = actual_winners

        # Predict.
        X = df_round[FEATURE_LIST]
        preds = model.predict(X)
        df_round = df_round.copy()
        df_round['Pred_Win__1'] = preds

        pred_teams = df_round['Team__1'].where(preds, df_round['Team__2'])
        pred_seeds = df_round['Seed__1'].where(preds, df_round['Seed__2'])

        pred_teams_by_round.append(pred_teams.tolist())
        pred_seeds_by_round.append(pred_seeds.tolist())
        prev_pred_teams = pred_teams.tolist()

        if not is_current:
            correct = (pred_teams == df_round['Winning_Team']).tolist()
            n_correct = sum(correct)
            round_score = n_correct * (2 ** (rnd - 1)) * 10
            correct_by_round.append(correct)
            num_correct_by_round.append(n_correct)
            total_score += round_score
            picks_str = '  '.join(
                f'[{pred_seeds.iloc[k]}]{pred_teams.iloc[k]} {"✓" if correct[k] else "✗"}'
                for k in range(len(pred_teams))
            )
            print(f'  Round {rnd} ({n_correct} correct, {round_score} pts): {picks_str}')
        else:
            correct_by_round.append([])
            num_correct_by_round.append(0)
            picks_str = '  '.join(
                f'[{pred_seeds.iloc[k]}]{pred_teams.iloc[k]}'
                for k in range(len(pred_teams))
            )
            print(f'  Round {rnd}: {picks_str}')

    return pred_teams_by_round, pred_seeds_by_round, correct_by_round, num_correct_by_round, total_score


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_pred_file(
    pred_teams_by_round,
    pred_seeds_by_round,
    correct_by_round,
    num_correct_by_round,
    total_score,
    is_current: bool,
) -> str:
    lines = []
    num_correct_total = 0
    for rnd_idx in range(6):
        parts = []
        n = len(pred_teams_by_round[rnd_idx])
        if not is_current:
            n_cor = num_correct_by_round[rnd_idx]
            round_score = n_cor * (2 ** rnd_idx) * 10
            num_correct_total += n_cor
            parts.append(f'{n_cor} for {n}')
            parts.append(str(round_score))
        for j in range(n):
            entry = f'[{pred_seeds_by_round[rnd_idx][j]}]{pred_teams_by_round[rnd_idx][j]}'
            if not is_current:
                entry += f'({int(correct_by_round[rnd_idx][j])})'
            parts.append(entry)
        lines.append(','.join(parts))
    result = '\n'.join(lines)
    if not is_current:
        result += f'\n{num_correct_total} for 63,{total_score}'
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Predict NCAA brackets using a single trained model evaluated across all years.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--model', '-m',
        default='logistic_lbfgs',
        choices=list(MODEL_REGISTRY),
        help='Model to use for predictions.',
    )
    parser.add_argument(
        '--data-root', '-d',
        default=str(Path(__file__).resolve().parents[1]),
        help='Path to repo root (contains Data/ directory).',
    )
    parser.add_argument(
        '--output-root', '-o',
        default=str(Path(__file__).resolve().parents[1]),
        help='Path under which Predictions/<model>/ outputs are written.',
    )
    parser.add_argument(
        '--this-year',
        type=int,
        default=None,
        help=(
            'Treat this year as the "current" year: its bracket is predicted but not scored. '
            'If omitted, all years are treated as historical and fully scored.'
        ),
    )
    parser.add_argument(
        '--final-four-pairings',
        default='0-1,2-3',
        help=(
            'How the 4 predicted Elite Eight winners (indexed 0-3 in CSV order) '
            'are paired for the Final Four.  Used ONLY for the current year; '
            'past years derive pairings from actual Round 5 data.  '
            'Format: "i-j,k-l", e.g. "0-2,1-3".  Default: "0-1,2-3".'
        ),
    )
    args = parser.parse_args()

    data_root   = Path(args.data_root)
    output_root = Path(args.output_root) / 'Predictions' / args.model
    output_root.mkdir(parents=True, exist_ok=True)

    this_year = args.this_year
    current_year_ff_pairings = parse_ff_pairings_arg(args.final_four_pairings)

    # Build the list of years to process: all completed years, plus the current
    # year appended at the end if it was supplied and isn't already in ALL_YEARS.
    years_to_process = list(ALL_YEARS)
    if this_year is not None and this_year not in years_to_process:
        years_to_process.append(this_year)
    num_eval_years = len(years_to_process) - (1 if this_year is not None else 0)

    # -----------------------------------------------------------------------
    # Train model once on a random train/test split of all historical data.
    # -----------------------------------------------------------------------
    df_all = load_combined_games(data_root)
    X_all, y_all = df_all[FEATURE_LIST], df_all['Win__1']
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.33, random_state=42)
    model = build_and_train_model(args.model, X_tr, y_tr)
    print(f'Model: {args.model}')
    print(f'  Train accuracy: {model.score(X_tr, y_tr):.4f}')
    print(f'  Test  accuracy: {model.score(X_te, y_te):.4f}')

    # -----------------------------------------------------------------------
    # Per-year loop
    # -----------------------------------------------------------------------
    total_correct_by_round = [0] * 7   # index 0 unused; rounds 1-6 at [1]-[6]
    total_score = 0

    for year in years_to_process:
        print(f'\n{"="*50}\n{year}\n{"="*50}')
        is_current = (this_year is not None and year == this_year)

        # Determine Final Four pairings.
        if is_current:
            ff_pairings = current_year_ff_pairings
        else:
            try:
                ff_pairings = derive_ff_pairings_from_data(data_root, year)
            except Exception as e:
                print(f'  WARNING: could not derive FF pairings ({e}), using default 0-1,2-3')
                ff_pairings = [(0, 1), (2, 3)]

        print(f'  FF pairings (R4 indices): {ff_pairings}')

        # Simulate.
        pred_teams, pred_seeds, correct, n_correct, score = simulate_bracket(
            model=model,
            data_root=data_root,
            year=year,
            this_year=this_year,
            ff_pairings=ff_pairings,
        )

        if not is_current:
            for rnd in range(1, 7):
                total_correct_by_round[rnd] += n_correct[rnd - 1]
            total_score += score
            print(f'  Year total: {sum(n_correct)} for 63, {score} pts')

        # Write prediction file.
        pred_str = format_pred_file(pred_teams, pred_seeds, correct, n_correct, score, is_current)
        out_path = output_root / f'{year}.csv'
        out_path.write_text(pred_str)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    games_per_round = [32, 16, 8, 4, 2, 1]
    summary_lines = ['OVERALL PERFORMANCE']
    for rnd in range(1, 7):
        total_games = games_per_round[rnd - 1] * num_eval_years
        n = total_correct_by_round[rnd]
        pct = n / total_games * 100 if total_games else 0
        pts = n * (2 ** (rnd - 1)) * 10
        summary_lines.append(
            f'  Round {rnd}: {n}/{total_games} ({pct:.1f}%), {pts} pts'
        )
    total_games_all = 63 * num_eval_years
    total_correct_all = sum(total_correct_by_round)
    summary_lines.append(
        f'  All rounds: {total_correct_all}/{total_games_all} '
        f'({total_correct_all / total_games_all * 100:.1f}%)'
    )
    summary_lines.append(f'  Avg bracket score: {total_score / num_eval_years:.1f}')
    summary_lines.append(f'  Train accuracy: {model.score(X_tr, y_tr):.4f}')
    summary_lines.append(f'  Test  accuracy: {model.score(X_te, y_te):.4f}')

    summary_str = '\n'.join(summary_lines)
    print(f'\n{summary_str}')
    (output_root / 'summary.txt').write_text(summary_str)


if __name__ == '__main__':
    main()
