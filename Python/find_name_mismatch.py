import pandas as pd

games_mismatch_names = []
kp_mismatch_names = []
for year in range(2024, 2025):
    if year == 2020:
        continue
    # df_games = pd.read_csv('../Data/GameData/' + str(year) + '.csv')
    df_games = pd.read_csv('../Data/BracketData/2024/Round1_' + str(year) + '.csv')
    df_kp = pd.read_csv('../Data/KenPomData/' + str(year) + '.csv')
    # print(df_bracket)
    # print(df_kp['Team'].values)
    for index, row in df_games.iterrows():
        team1 = row['Team__1']
        # print(team1)
        # print(df_kp.loc[df_kp['Team'] == team1])
        if team1 not in df_kp['Team'].values and \
           team1 not in games_mismatch_names:
            games_mismatch_names.append(team1)

        team2 = row['Team__2']
        # print(team2)
        # print(df_kp.loc[df_kp['Team'] == team2])
        # print()
        if team2 not in df_kp['Team'].values and \
           team2 not in games_mismatch_names:
            games_mismatch_names.append(team2)

    for index, row in df_kp.iterrows():
        team = row['Team']
        if team not in df_games['Team__1'].values and \
           team not in df_games['Team__2'].values and \
           team not in kp_mismatch_names:
            kp_mismatch_names.append(team)

print('TEAM NAMES IN GAMES BUT NOT IN KENPOM (length=' + str(len(games_mismatch_names)) + '):')
print(pd.Series(games_mismatch_names).sort_values())
print()
print('TEAM NAMES IN KENPOM BUT NOT IN GAMES (length=' + str(len(kp_mismatch_names)) + '):')
print(pd.Series(kp_mismatch_names).sort_values())
