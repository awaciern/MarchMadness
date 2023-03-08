import pandas as pd
from sklearn.linear_model import LogisticRegression
import csv

YEAR = 2022
FINAL_FOUR_SETUP = [0, 1, 2, 3]

df_games = pd.read_csv('..\\Data\\GameCombinedData\\All.csv')

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
print(X_train.shape)

y = df_games['Win__1']
print(y.shape)

model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000).fit(X_train, y)

print(model.score(X_train, y))
print(type(model.predict(X_train)))

pred_teams_by_round = []
pred_was_correct_by_round = []
pred_seeds_by_round = []
num_correct_by_round = []
score_by_round = []
score = 0
for round in range(1, 7):
    df_round = pd.read_csv('..\\Data\\BracketCombinedData\\' + str(YEAR) + '\\Round'
                           + str(round) + '_' + str(YEAR) + '.csv')
    if round != 1:
        winning_teams = df_round['Winning_Team']
        # print(winning_teams)
        df_round = pd.DataFrame(columns=['Team__1', 'Team__2'])
        if round == 5:
            pred_teams = [pred_teams[FINAL_FOUR_SETUP[i]] for i in range(0, 4)]
            print(pred_teams)
        for i in range(0, len(pred_teams), 2):
            # print(pred_teams[i])
            # print(pred_teams[i+1])
            df_round.loc[i//2] = [pred_teams[i], pred_teams[i+1]]
        # print(df_round)

        df_kp = pd.read_csv('..\\Data\\KenPomData\\' + str(YEAR) + '.csv')

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

        df_round['Winning_Team'] = winning_teams
        # print(df_round)

    print(round)
    X_round = df_round[feat_list]
    y_round = model.predict(X_round)
    df_round['Pred_Win__1'] = y_round
    # print(df_round)

    pred_teams = df_round['Team__1'].where(df_round.Pred_Win__1 == True, df_round['Team__2'])
    pred_teams_by_round.append(pred_teams.to_list())
    print(pred_teams)

    pred_seeds = df_round['Seed__1'].where(df_round.Pred_Win__1 == True, df_round['Seed__2'])
    pred_seeds_by_round.append(pred_seeds.to_list())
    print(pred_teams)

    pred_was_correct = pred_teams == df_round['Winning_Team']
    pred_was_correct_by_round.append(pred_was_correct.to_list())
    num_correct = pred_was_correct.sum()
    num_correct_by_round.append(num_correct)
    print(num_correct)
    round_score = num_correct * 2**(round-1) * 10
    print(round_score)
    score_by_round.append(round_score)
    score += round_score
    print()

print(score)
print(num_correct_by_round)
print(score_by_round)
print(pred_teams_by_round)
print(pred_seeds_by_round)
print(pred_was_correct_by_round)

pred_file_str = ''
num_correct_total = 0
for i in range(0, 6):
    num_correct_total += num_correct_by_round[i]
    num_picks = len(pred_teams_by_round[i])
    pred_file_str += str(num_correct_by_round[i]) + ' for ' + str(num_picks) + ','
    pred_file_str += str(score_by_round[i]) + ','
    for j in range(0, num_picks):
        pred_file_str += '[' + str(pred_seeds_by_round[i][j]) + ']' + \
                         pred_teams_by_round[i][j] + \
                         '(' + str(int(pred_was_correct_by_round[i][j])) + '),'
    pred_file_str = pred_file_str[:-1]
    pred_file_str += '\n'
pred_file_str += str(num_correct_total) + ' for 63,'
pred_file_str += str(score)

print(pred_file_str)
with open('..\\Predictions\\LogisticPOC\\' + str(YEAR) + '.csv', 'w') as pred_file:
    pred_file.write(pred_file_str)
