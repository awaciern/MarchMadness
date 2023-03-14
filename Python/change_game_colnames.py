import pandas as pd

COL_NAME_CHANGE_DICT = {}
COL_NAME_CHANGE_DICT['Team1'] = 'Team__1'
COL_NAME_CHANGE_DICT['Team1_Seed'] = 'Seed__1'
# COL_NAME_CHANGE_DICT['Team1_Score'] = 'Score__1'
COL_NAME_CHANGE_DICT['Team2'] = 'Team__2'
COL_NAME_CHANGE_DICT['Team2_Seed'] = 'Seed__2'
# COL_NAME_CHANGE_DICT['Team2_Score'] = 'Score__2'
# COL_NAME_CHANGE_DICT['WinningTeam'] = 'Winning_Team'
# COL_NAME_CHANGE_DICT['Team1_Win'] = 'Win__1'
COL_NAME_CHANGE_DICT['Year'] = 'Year'
COL_NAME_CHANGE_DICT['Round'] = 'Round'

for year in range(2023, 2024):
    df_year_games = pd.DataFrame()
    if year == 2020:
        continue

    # df_year_games = pd.read_csv('..\\Data\\GameData\\' + str(year) + '.csv')
    # for key, value in COL_NAME_CHANGE_DICT.items():
    #     df_year_games.rename(columns={key : value}, inplace=True)
    # df_year_games.to_csv(path_or_buf='..\\Data\\GameData\\' + str(year) + '.csv', index=False)

    for round in range(1, 2):
        df_round = pd.read_csv('..\\Data\\BracketData\\' + str(year) + '\\Round' +
                               str(round) + '_' + str(year) + '.csv')
        for key, value in COL_NAME_CHANGE_DICT.items():
            if key == 'Year' or key == 'Round':
                continue
            df_round.rename(columns={key : value}, inplace=True)
        df_round.to_csv(path_or_buf='..\\Data\\BracketData\\' + str(year) + '\\Round' +
                        str(round) + '_' + str(year) + '.csv', index=False)

# df_all_games = pd.read_csv('..\\Data\\GameData\\All.csv')
# for key, value in COL_NAME_CHANGE_DICT.items():
#     df_all_games.rename(columns={key : value}, inplace=True)
# df_all_games.to_csv(path_or_buf='..\\Data\\GameData\\All.csv', index=False)
