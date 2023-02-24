import pandas as pd

# j = 0
df_all_games = pd.DataFrame()
for year in range(2012, 2023):
    df_year_games = pd.DataFrame()
    if year == 2020:
        continue
    for round in range(1, 7):
        # i = 0
        df = pd.read_csv('..\\Data\\BracketData\\' + str(year) + '\\Round' +
                         str(round) + '_' + str(year) + '.csv')
        # i += len(df)
        # j += len(df)
        # print(df)
        # print('YEAR: ' + str(year) + ', ROUND: ' + str(round) + ', i = ' + str(i))
        df_all_games = pd.concat([df_all_games, df])
        df_year_games = pd.concat([df_year_games, df])
    df_year_games.to_csv(path_or_buf='..\\Data\\GameData\\' + str(year) + '.csv', index=False)

print(df_all_games)
df_all_games.to_csv(path_or_buf='..\\Data\\GameData\\All.csv', index=False)
# print(j)
