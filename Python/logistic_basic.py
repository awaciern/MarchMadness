import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('..\\Data\\GameCombinedData\\All.csv')

X = df[[
        # 'Team__1',
        'Seed__1',
        'Rk_AdjEM__1',
        # 'Conf__1',
        'Wins__1',
        'Losses__1',
        'WinPct__1',
        'AdjEM__1',
        'AdjO__1',
        'Rk_AdjO__1',
        'AdjD__1',
        'Rk_AdjD__1',
        'AdjT__1',
        'Rk_AdjT__1',
        'Luck__1',
        'Rk_Luck__1',
        'SOS_AdjEM__1',
        'Rk_SOS_AdjEM__1',
        'SOS_AdjO__1',
        'Rk_SOS_AdjO__1',
        'SOS_AdjD__1',
        'Rk_SOS_AdjD__1',
        # 'NCSOS_AdjEM__1',
        'Rk_NCSOS_AdjEM__1',
        # 'Team__2',
        'Seed__2',
        'Rk_AdjEM__2',
        # 'Conf__2',
        'Wins__2',
        'Losses__2',
        'WinPct__2',
        'AdjEM__2',
        'AdjO__2',
        'Rk_AdjO__2',
        'AdjD__2',
        'Rk_AdjD__2',
        'AdjT__2',
        'Rk_AdjT__2',
        'Luck__2',
        'Rk_Luck__2',
        'SOS_AdjEM__2',
        'Rk_SOS_AdjEM__2',
        'SOS_AdjO__2',
        'Rk_SOS_AdjO__2',
        'SOS_AdjD__2',
        'Rk_SOS_AdjD__2',
        # 'NCSOS_AdjEM__2',
        'Rk_NCSOS_AdjEM__2',
        'Round',
        ]]
# X.dropna(inplace=True)
print(X.shape)

y = df['Win__1']
print(y.shape)

model = LogisticRegression(random_state=0, solver='sag', max_iter=10000).fit(X, y)

print(model.score(X, y))
