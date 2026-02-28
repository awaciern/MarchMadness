import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Find team name mismatches between bracket and KenPom data.')
parser.add_argument('bracket_files', nargs='+', help='One or more bracket CSV files (e.g. Round1_2025.csv Round2_2025.csv)')
parser.add_argument('kenpom_file', help='KenPom CSV file (e.g. ../Data/KenPomData/2025.csv)')
args = parser.parse_args()

games_mismatch_names = []
kp_mismatch_names = []

df_kp = pd.read_csv(args.kenpom_file)

for bracket_file in args.bracket_files:
    df_games = pd.read_csv(bracket_file)

    for index, row in df_games.iterrows():
        team1 = row['Team1']
        if team1 not in df_kp['Team'].values and \
           team1 not in games_mismatch_names:
            games_mismatch_names.append(team1)

        team2 = row['Team2']
        if team2 not in df_kp['Team'].values and \
           team2 not in games_mismatch_names:
            games_mismatch_names.append(team2)

    for index, row in df_kp.iterrows():
        team = row['Team']
        if team not in df_games['Team1'].values and \
           team not in df_games['Team2'].values and \
           team not in kp_mismatch_names:
            kp_mismatch_names.append(team)

print('TEAM NAMES IN GAMES BUT NOT IN KENPOM (length=' + str(len(games_mismatch_names)) + '):')
print(pd.Series(games_mismatch_names).sort_values())
print()
print('TEAM NAMES IN KENPOM BUT NOT IN GAMES (length=' + str(len(kp_mismatch_names)) + '):')
print(pd.Series(kp_mismatch_names).sort_values())
