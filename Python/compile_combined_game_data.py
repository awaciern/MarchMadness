import pandas as pd

df_all_games_combined = pd.DataFrame()
for year in range(2012, 2024):
    if year == 2020:
        continue

    print(year)

    df_games =  pd.read_csv('../Data/GameData/' + str(year) + '.csv')
    df_kp = pd.read_csv('../Data/KenPomData/' + str(year) + '.csv')

    for col_name in df_kp.columns:
        df_kp.rename(columns={col_name : col_name + '__1'}, inplace=True)
    df_join1 = df_games.join(other=df_kp.set_index(['Team__1', 'Seed__1']),
                             on=['Team__1', 'Seed__1'],
                             how='inner')
    print(df_join1.shape)

    for col_name in df_kp.columns:
        df_kp.rename(columns={col_name : col_name.replace('__1', '__2')}, inplace=True)
    df_join2 = df_games.join(other=df_kp.set_index(['Team__2', 'Seed__2']),
                             on=['Team__2', 'Seed__2'],
                             how='inner')
    print(df_join2.shape)

    join_cols_common = ['Team__1', 'Seed__1', 'Score__1',
                        'Team__2', 'Seed__2', 'Score__2',
                        'Winning_Team', 'Win__1', 'Year', 'Round']
    df_year_games_combined = df_join1.join(other=df_join2.set_index(join_cols_common),
                                           on=join_cols_common,
                                           how='left')

    cols = df_year_games_combined.columns.tolist()
    cols_new_order = []
    for col in cols:
        if '__1' in col and col != 'Win__1':
            cols_new_order.append(col)
    for col in cols:
        if '__2' in col:
            cols_new_order.append(col)
    for col in cols:
        if not ('__1' in col and col != 'Win__1') and '__2' not in col:
            cols_new_order.append(col)
    # print(cols_new_order)
    # print(len(cols_new_order))
    df_year_games_combined = df_year_games_combined[cols_new_order]

    # print(df_year_games_combined)
    print(df_year_games_combined.shape)
    print()

    df_year_games_combined.to_csv(path_or_buf='../Data/GameCombinedData/' + str(year) + '.csv', index=False)
    df_all_games_combined = pd.concat([df_all_games_combined, df_year_games_combined])

df_all_games_combined.to_csv(path_or_buf='../Data/GameCombinedData/All.csv', index=False)
print('ALL')
print(df_all_games_combined.shape)
