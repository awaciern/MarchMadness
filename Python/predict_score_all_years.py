import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import csv
import os

NUM_PAST_YEARS = 11
THIS_YEAR = 2024

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


df_games = pd.read_csv('../Data/GameCombinedData/All.csv')

feat_list = [
              # 'Team__1',
              'Seed__1',
              # 'Rk_AdjEM__1',
              # 'Conf__1',
              # 'Wins__1',
              # 'Losses__1',
              'WinPct__1',
              'AdjEM__1',
              'AdjO__1',
              # 'Rk_AdjO__1',
              'AdjD__1',
              # 'Rk_AdjD__1',
              'AdjT__1',
              # 'Rk_AdjT__1',
              'Luck__1',
              # 'Rk_Luck__1',
              'SOS_AdjEM__1',
              # 'Rk_SOS_AdjEM__1',
              'SOS_AdjO__1',
              # 'Rk_SOS_AdjO__1',
              'SOS_AdjD__1',
              # 'Rk_SOS_AdjD__1',
              # 'NCSOS_AdjEM__1',
              'Rk_NCSOS_AdjEM__1',
              # 'Team__2',
              'Seed__2',
              'Rk_AdjEM__2',
              # 'Conf__2',
              # 'Wins__2',
              # 'Losses__2',
              'WinPct__2',
              'AdjEM__2',
              'AdjO__2',
              # 'Rk_AdjO__2',
              'AdjD__2',
              # 'Rk_AdjD__2',
              'AdjT__2',
              # 'Rk_AdjT__2',
              'Luck__2',
              # 'Rk_Luck__2',
              'SOS_AdjEM__2',
              # 'Rk_SOS_AdjEM__2',
              'SOS_AdjO__2',
              # 'Rk_SOS_AdjO__2',
              'SOS_AdjD__2',
              # 'Rk_SOS_AdjD__2',
              # 'NCSOS_AdjEM__2',
              # 'Rk_NCSOS_AdjEM__2',
              # 'Round',
            ]

X_train = df_games[feat_list]
# print(X_train.shape)

y = df_games['Win__1']
# print(y.shape)

# MODEL = 'Logistic'
# model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000).fit(X_train, y)
# MODEL = 'KNN'
# model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y)
# MODEL = 'SVC_RBF'
# model = SVC(gamma='auto').fit(X_train, y)
# MODEL = 'SVC_Linear'
# model = SVC(kernel='linear').fit(X_train, y)
# MODEL = 'DecisionTree'
# model = DecisionTreeClassifier().fit(X_train, y)
# MODEL = 'RandomForest'
# model = RandomForestClassifier().fit(X_train, y)
# MODEL = 'AdaBoost'
# model = AdaBoostClassifier().fit(X_train, y)
MODEL = 'GpClass'
model = GaussianProcessClassifier().fit(X_train, y)

print(model.score(X_train, y))
# print(type(model.predict(X_train)))

total_num_correct_by_round = [0 for i in range(0, 7)]
total_score = 0
for year in range(2012, 2025):
    if year == 2020:
        continue

    print(str(year))

    pred_teams_by_round = []
    pred_seeds_by_round = []
    if year != THIS_YEAR:
        pred_was_correct_by_round = []
        num_correct_by_round = []
        score_by_round = []
        score = 0
    for round in range(1, 7):
        if year != THIS_YEAR or round == 1:
            df_round = pd.read_csv('../Data/BracketCombinedData/' + str(year) + '/Round'
                                    + str(round) + '_' + str(year) + '.csv')

        if round != 1:
            if year != THIS_YEAR:
                winning_teams = df_round['Winning_Team']
                # print(winning_teams)
            df_round = pd.DataFrame(columns=['Team__1', 'Team__2'])
            if round == 5:
                pred_teams = [pred_teams[FINAL_FOUR_SETUP_DICT[year][i]] for i in range(0, 4)]
                # print(pred_teams)
            for i in range(0, len(pred_teams), 2):
                # print(pred_teams[i])
                # print(pred_teams[i+1])
                df_round.loc[i//2] = [pred_teams[i], pred_teams[i+1]]
            # print(df_round)

            df_kp = pd.read_csv('../Data/KenPomData/' + str(year) + '.csv')

            for col_name in df_kp.columns:
                df_kp.rename(columns={col_name : col_name + '__1'}, inplace=True)
            df_join1 = df_round.join(other=df_kp.set_index(['Team__1']),
                                    on=['Team__1'],
                                    how='inner')
            # print(df_join1.shape)
            # print(df_join1)

            for col_name in df_kp.columns:
                df_kp.rename(columns={col_name : col_name.replace('__1', '__2')}, inplace=True)
            df_join2 = df_round.join(other=df_kp.set_index(['Team__2']),
                                    on=['Team__2'],
                                    how='inner')
            # print(df_join2.shape)
            # print(df_join2)

            join_cols_common = ['Team__1', 'Team__2']
            df_round = df_join1.join(other=df_join2.set_index(join_cols_common),
                                    on=join_cols_common,
                                    how='inner')
            # print(df_round.shape)
            # print(df_round)

            if year != THIS_YEAR:
                df_round['Winning_Team'] = winning_teams
            # print(df_round)

        # print(round)
        # print(df_round)
        X_round = df_round[feat_list]
        y_round = model.predict(X_round)
        df_round['Pred_Win__1'] = y_round
        # print(df_round)

        pred_teams = df_round['Team__1'].where(df_round.Pred_Win__1 == True, df_round['Team__2'])
        pred_teams_by_round.append(pred_teams.to_list())
        # print(pred_teams)

        pred_seeds = df_round['Seed__1'].where(df_round.Pred_Win__1 == True, df_round['Seed__2'])
        pred_seeds_by_round.append(pred_seeds.to_list())
        # print(pred_teams)

        if year != THIS_YEAR:
            pred_was_correct = pred_teams == df_round['Winning_Team']
            pred_was_correct_by_round.append(pred_was_correct.to_list())
            num_correct = pred_was_correct.sum()
            num_correct_by_round.append(num_correct)
            total_num_correct_by_round[round] += num_correct
            # print(num_correct)
            round_score = num_correct * 2**(round-1) * 10
            # print(round_score)
            score_by_round.append(round_score)
            score += round_score
            print(str(round) + ': ' + str(num_correct) + ", " + str(round_score))

    if year != THIS_YEAR:
        print(score)
        print(num_correct_by_round)
        print(score_by_round)
        # print(pred_teams_by_round)
        # print(pred_seeds_by_round)
        # print(pred_was_correct_by_round)

    pred_file_str = ''
    if year != THIS_YEAR:
        num_correct_total = 0
    for i in range(0, 6):
        num_picks = len(pred_teams_by_round[i])
        if year != THIS_YEAR:
            num_correct_total += num_correct_by_round[i]
            pred_file_str += str(num_correct_by_round[i]) + ' for ' + str(num_picks) + ','
            pred_file_str += str(score_by_round[i]) + ','
        for j in range(0, num_picks):
            pred_file_str += '[' + str(pred_seeds_by_round[i][j]) + ']' + \
                            pred_teams_by_round[i][j] + ','
            if year != THIS_YEAR:
                pred_file_str += '(' + str(int(pred_was_correct_by_round[i][j])) + '),'
        pred_file_str = pred_file_str[:-1]
        pred_file_str += '\n'
    if year != THIS_YEAR:
        pred_file_str += str(num_correct_total) + ' for 63,'
        pred_file_str += str(score)
        total_score += score

    if not os.path.exists('../Predictions/' + MODEL):
        os.mkdir('../Predictions/' + MODEL)

    print(pred_file_str)
    with open('../Predictions/' + MODEL + '/' + str(year) + '.csv', 'w') as pred_file:
        pred_file.write(pred_file_str)
    print()

summary_str = 'OVERALL PERFORMANCE\n'
summary_str += f'ROUND 1: {total_num_correct_by_round[1]} for {32 * NUM_PAST_YEARS}, {total_num_correct_by_round[1] / (32 * NUM_PAST_YEARS) * 100:.2f}%, {10 * total_num_correct_by_round[1]} points\n'
summary_str += f'ROUND 2: {total_num_correct_by_round[2]} for {16 * NUM_PAST_YEARS}, {total_num_correct_by_round[2] / (16 * NUM_PAST_YEARS) * 100:.2f}%, {20 * total_num_correct_by_round[2]} points\n'
summary_str += f'ROUND 3: {total_num_correct_by_round[3]} for {8 * NUM_PAST_YEARS}, {total_num_correct_by_round[3] / (8 * NUM_PAST_YEARS) * 100:.2f}%, {40 * total_num_correct_by_round[3]} points\n'
summary_str += f'ROUND 4: {total_num_correct_by_round[4]} for {4 * NUM_PAST_YEARS}, {total_num_correct_by_round[4] / (4 * NUM_PAST_YEARS) * 100:.2f}%, {80 * total_num_correct_by_round[4]} points\n'
summary_str += f'ROUND 5: {total_num_correct_by_round[5]} for {2 * NUM_PAST_YEARS}, {total_num_correct_by_round[5] / (2 * NUM_PAST_YEARS) * 100:.2f}%, {160 * total_num_correct_by_round[5]} points\n'
summary_str += f'ROUND 6: {total_num_correct_by_round[6]} for {1 * NUM_PAST_YEARS}, {total_num_correct_by_round[6] / (1 * NUM_PAST_YEARS) * 100:.2f}%, {320 * total_num_correct_by_round[6]} points\n'
summary_str += f'ALL ROUNDS: {sum(total_num_correct_by_round)} for {63 * NUM_PAST_YEARS }, {sum(total_num_correct_by_round) / (63 * NUM_PAST_YEARS) * 100:.2f}%\n'
summary_str += f'AVG BRACKET SCORE = {total_score / NUM_PAST_YEARS:.2f}\n'
print(summary_str)
with open('../Predictions/' + MODEL + '/summary.txt', 'w') as file:
    file.write(summary_str)