import argparse
import os
from pathlib import Path
from typing import Callable, Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split

NUM_PAST_YEARS = 12
THIS_YEAR = 2025

FINAL_FOUR_SETUP_DICT = {}
FINAL_FOUR_SETUP_DICT[2012] = [2, 0, 1, 3]
FINAL_FOUR_SETUP_DICT[2013] = [3, 0, 2, 1]
FINAL_FOUR_SETUP_DICT[2014] = [3, 0, 1, 2]
FINAL_FOUR_SETUP_DICT[2015] = [3, 0, 1, 2]
FINAL_FOUR_SETUP_DICT[2016] = [2, 0, 1, 3]
FINAL_FOUR_SETUP_DICT[2017] = [0, 1, 3, 2]
FINAL_FOUR_SETUP_DICT[2018] = [2, 0, 1, 3]
FINAL_FOUR_SETUP_DICT[2019] = [1, 0, 2, 3]
FINAL_FOUR_SETUP_DICT[2021] = [0, 1, 2, 3]
FINAL_FOUR_SETUP_DICT[2022] = [0, 1, 2, 3]
FINAL_FOUR_SETUP_DICT[2023] = [2, 1, 3, 0]
FINAL_FOUR_SETUP_DICT[2024] = [1, 0, 2, 3]
FINAL_FOUR_SETUP_DICT[2025] = [1, 0, 2, 3]

FEATURE_LIST = [
    'WinPct__1', 'AdjEM__1', 'AdjO__1', 'AdjD__1', 'AdjT__1', 'Luck__1',
    'SOS_AdjEM__1', 'Rk_NCSOS_AdjEM__1', 'Rk_AdjEM__2', 'WinPct__2',
    'AdjEM__2', 'AdjO__2', 'AdjD__2', 'AdjT__2', 'Luck__2',
    'SOS_AdjEM__2', 'Rk_NCSOS_AdjEM__2',
]


def get_model_registry() -> Dict[str, Callable[[], object]]:
    """Return a registry mapping model keys to constructors (unfitted estimators)."""
    return {
        'logistic_lbfgs': lambda: LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000),
        'logistic_newton': lambda: LogisticRegression(random_state=0, solver='newton-cg', max_iter=1000),
        'logistic_liblinear': lambda: LogisticRegression(random_state=0, solver='liblinear', max_iter=1000),
        'knn5': lambda: KNeighborsClassifier(n_neighbors=5),
        'svc_rbf': lambda: SVC(gamma='auto'),
        'svc_linear': lambda: SVC(kernel='linear'),
        'svc_poly2': lambda: SVC(kernel='poly', degree=2),
        'svc_poly3': lambda: SVC(kernel='poly', degree=3),
        'decision_tree': lambda: DecisionTreeClassifier(),
        'random_forest': lambda: RandomForestClassifier(),
        'adaboost': lambda: AdaBoostClassifier(),
        'gp': lambda: GaussianProcessClassifier(),
    }


def load_game_data(base_path: Path) -> pd.DataFrame:
    """Load combined game data used for training."""
    path = base_path / 'Data' / 'GameCombinedData' / 'All.csv'
    return pd.read_csv(path)


def train_model_by_key(model_key: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Instantiate and fit a model selected by key."""
    registry = get_model_registry()
    if model_key not in registry:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(registry.keys())}")
    estimator = registry[model_key]()
    estimator.fit(X_train, y_train)
    return estimator


def ensure_output_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def predict_and_score_all_years(
    model,
    model_name: str,
    data_root: Path,
    output_root: Path,
    years: range,
    feature_list,
    this_year: int,
    final_four_setup: dict
):
    """Main loop: predict brackets for each year, compute scoring stats, and write outputs."""
    total_num_correct_by_round = [0 for _ in range(0, 7)]
    total_score = 0

    predictions_dir = output_root / model_name
    ensure_output_dir(predictions_dir)

    for year in years:
        if year == 2020:
            continue
        print(year)

        pred_teams_by_round = []
        pred_seeds_by_round = []
        if year != this_year:
            pred_was_correct_by_round = []
            num_correct_by_round = []
            score_by_round = []
            score = 0

        for rnd in range(1, 7):
            # Load initial round CSV for past years or for round 1 of this year.
            if year != this_year or rnd == 1:
                round_path = data_root / 'Data' / 'BracketCombinedData' / str(year) / f'Round{rnd}_{year}.csv'
                df_round = pd.read_csv(round_path)

            if rnd != 1:
                if year != this_year:
                    winning_teams = df_round['Winning_Team']

                # build next-round pairing frame with Team__1 and Team__2
                df_round = pd.DataFrame(columns=['Team__1', 'Team__2'])
                if rnd == 5:
                    # reorder final four according to historical setup mapping
                    pred_teams = [pred_teams[final_four_setup[year][i]] for i in range(4)]

                for i in range(0, len(pred_teams), 2):
                    df_round.loc[i // 2] = [pred_teams[i], pred_teams[i + 1]]

                # Join KenPom stats for Team__1 and Team__2
                df_kp = pd.read_csv(data_root / 'Data' / 'KenPomData' / f'{year}.csv')
                # join team 1 columns
                df_kp_1 = df_kp.add_suffix('__1').rename(columns={'Team__1': 'Team__1'})
                df_join1 = df_round.join(other=df_kp_1.set_index(['Team__1']), on=['Team__1'], how='inner')

                # prepare team 2 columns then join
                df_kp_2 = df_kp.add_suffix('__2').rename(columns={'Team__2': 'Team__2'})
                df_join2 = df_round.join(other=df_kp_2.set_index(['Team__2']), on=['Team__2'], how='inner')

                # combine both joins on Team__1 and Team__2
                join_cols_common = ['Team__1', 'Team__2']
                # Prefer join produced earlier but re-join to match original shape
                df_round = df_join1.join(other=df_join2.set_index(join_cols_common), on=join_cols_common, how='inner')

                if year != this_year:
                    df_round['Winning_Team'] = winning_teams

            # run prediction for this round
            X_round = df_round[feature_list]
            preds = model.predict(X_round)
            df_round['Pred_Win__1'] = preds

            # collect predicted teams and seeds
            pred_teams = df_round['Team__1'].where(df_round.Pred_Win__1 == True, df_round['Team__2'])
            pred_teams_by_round.append(pred_teams.to_list())

            pred_seeds = df_round['Seed__1'].where(df_round.Pred_Win__1 == True, df_round['Seed__2'])
            pred_seeds_by_round.append(pred_seeds.to_list())

            if year != this_year:
                pred_was_correct = pred_teams == df_round['Winning_Team']
                pred_was_correct_by_round.append(pred_was_correct.to_list())
                num_correct = int(pred_was_correct.sum())
                num_correct_by_round.append(num_correct)
                total_num_correct_by_round[rnd] += num_correct
                round_score = num_correct * 2 ** (rnd - 1) * 10
                score_by_round.append(round_score)
                score += round_score
                print(f'Round {rnd}: {num_correct}, {round_score}')

        # assemble prediction output string (same layout as original)
        pred_file_str = ''
        if year != this_year:
            num_correct_total = 0
        for i in range(0, 6):
            num_picks = len(pred_teams_by_round[i])
            if year != this_year:
                num_correct_total += num_correct_by_round[i]
                pred_file_str += f'{num_correct_by_round[i]} for {num_picks},'
                pred_file_str += f'{score_by_round[i]},'
            for j in range(0, num_picks):
                pred_file_str += f'[{pred_seeds_by_round[i][j]}]{pred_teams_by_round[i][j]},'
                if year != this_year:
                    pred_file_str += f'({int(pred_was_correct_by_round[i][j])}),'
            pred_file_str = pred_file_str[:-1]
            pred_file_str += '\n'
        if year != this_year:
            pred_file_str += f'{num_correct_total} for 63,'
            pred_file_str += f'{score}'
            total_score += score

        print(pred_file_str)
        with open(predictions_dir / f'{year}.csv', 'w') as pred_file:
            pred_file.write(pred_file_str)
        print()

    # summary (same metrics as original)
    summary_lines = []
    # model scoring will be added by caller with access to train/test sets
    return total_num_correct_by_round, total_score


def main():
    parser = argparse.ArgumentParser(description='Predict NCAA brackets across years using scikit-learn models.')
    parser.add_argument('--model', '-m', default='logistic_lbfgs', help='Model key to use (see supported keys).')
    parser.add_argument('--data-root', '-d', default=str(Path(__file__).resolve().parents[1]), help='Root path containing Data/ directory')
    parser.add_argument('--output-root', '-o', default=str(Path(__file__).resolve().parents[1]), help='Root path for Predictions2/')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root) / 'Predictions2'

    # Load training data
    df_games = load_game_data(data_root)
    X = df_games[FEATURE_LIST]
    y = df_games['Win__1']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Train model selected via CLI
    model = train_model_by_key(args.model, X_train, y_train)

    # Run predictions / scoring for each year
    years = range(2012, 2026)
    total_num_correct_by_round, total_score = predict_and_score_all_years(
        model=model,
        model_name=args.model,
        data_root=data_root,
        output_root=output_root,
        years=years,
        feature_list=FEATURE_LIST,
        this_year=THIS_YEAR,
        final_four_setup=FINAL_FOUR_SETUP_DICT
    )

    # Print summary and save to file
    summary_str = 'OVERALL PERFORMANCE\n'
    summary_str += f'Train Score = {model.score(X_train, y_train)}\n'
    summary_str += f'Test Score = {model.score(X_test, y_test)}\n'
    summary_str += f'ROUND 1: {total_num_correct_by_round[1]} for {32 * NUM_PAST_YEARS}, {total_num_correct_by_round[1] / (32 * NUM_PAST_YEARS) * 100:.2f}%, {10 * total_num_correct_by_round[1]} points\n'
    summary_str += f'ROUND 2: {total_num_correct_by_round[2]} for {16 * NUM_PAST_YEARS}, {total_num_correct_by_round[2] / (16 * NUM_PAST_YEARS) * 100:.2f}%, {20 * total_num_correct_by_round[2]} points\n'
    summary_str += f'ROUND 3: {total_num_correct_by_round[3]} for {8 * NUM_PAST_YEARS}, {total_num_correct_by_round[3] / (8 * NUM_PAST_YEARS) * 100:.2f}%, {40 * total_num_correct_by_round[3]} points\n'
    summary_str += f'ROUND 4: {total_num_correct_by_round[4]} for {4 * NUM_PAST_YEARS}, {total_num_correct_by_round[4] / (4 * NUM_PAST_YEARS) * 100:.2f}%, {80 * total_num_correct_by_round[4]} points\n'
    summary_str += f'ROUND 5: {total_num_correct_by_round[5]} for {2 * NUM_PAST_YEARS}, {total_num_correct_by_round[5] / (2 * NUM_PAST_YEARS) * 100:.2f}%, {160 * total_num_correct_by_round[5]} points\n'
    summary_str += f'ROUND 6: {total_num_correct_by_round[6]} for {1 * NUM_PAST_YEARS}, {total_num_correct_by_round[6] / (1 * NUM_PAST_YEARS) * 100:.2f}%, {320 * total_num_correct_by_round[6]} points\n'
    summary_str += f'ALL ROUNDS: {sum(total_num_correct_by_round)} for {63 * NUM_PAST_YEARS }, {sum(total_num_correct_by_round) / (63 * NUM_PAST_YEARS) * 100:.2f}%\n'
    summary_str += f'AVG BRACKET SCORE = {total_score / NUM_PAST_YEARS:.2f}\n'
    print(summary_str)

    out_dir = Path(args.output_root) / 'Predictions2' / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'summary.txt', 'w') as f:
        f.write(summary_str)


if __name__ == '__main__':
    main()