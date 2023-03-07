import pandas as pd
import os

df_all_games_combined = pd.DataFrame()
for year in range(2012, 2023):
    if year == 2020:
        continue

    print(year)

    df_kp = pd.read_csv('..\\Data\\KenPomData\\' + str(year) + '.csv')

    df_kp1 = df_kp.copy()
    for col_name in df_kp1.columns:
       df_kp1.rename(columns={col_name : col_name + '__1'}, inplace=True)

    df_kp2 = df_kp.copy()
    for col_name in df_kp2.columns:
       df_kp2.rename(columns={col_name : col_name + '__2'}, inplace=True)

    for round in range(1, 7):
        df_round = pd.read_csv('..\\Data\\BracketData\\' + str(year) + '\\Round' +
                               str(round) + '_' + str(year) + '.csv')

        df_join1 = df_round.join(other=df_kp1.set_index(['Team__1', 'Seed__1']),
                                 on=['Team__1', 'Seed__1'],
                                 how='inner')
        print(df_join1.shape)

        df_join2 = df_round.join(other=df_kp2.set_index(['Team__2', 'Seed__2']),
                                 on=['Team__2', 'Seed__2'],
                                 how='inner')
        print(df_join1.shape)

        join_cols_common = ['Team__1', 'Seed__1', 'Score__1',
                            'Team__2', 'Seed__2', 'Score__2',
                            'Winning_Team', 'Win__1']
        df_round_combined = df_join1.join(other=df_join2.set_index(join_cols_common),
                                          on=join_cols_common,
                                          how='left')

        cols = df_round_combined.columns.tolist()
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
        df_round_combined = df_round_combined[cols_new_order]
        print(df_round_combined.shape)
        print()

        if not os.path.exists('..\\Data\\BracketCombinedData\\' + str(year)):
            os.mkdir('..\\Data\\BracketCombinedData\\' + str(year))

        df_round_combined.to_csv(path_or_buf='..\\Data\\BracketCombinedData\\' + str(year) +
                                 '\\Round' + str(round) + '_' + str(year) + '.csv', index=False)
