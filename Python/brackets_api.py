import argparse
import requests
import csv
import os

parser = argparse.ArgumentParser(description='Fetch NCAA bracket data for a given year.')
parser.add_argument('year', type=str, help='The tournament year (e.g. 2024)')
args = parser.parse_args()

YEAR = args.year
URL = f'https://ncaa-api.henrygd.me/brackets/basketball-men/d1/{YEAR}'

response = requests.get(URL)
response.raise_for_status()
data = response.json()

championships = data.get('championships', [])
if not championships:
    print("No championship data found")
    exit()

championship = championships[0]
games = championship['games']

# bracketPositionId ranges per round (First Four is 101-104, excluded)
# Round 1 = Round of 64:  201-232
# Round 2 = Round of 32:  301-316
# Round 3 = Sweet 16:     401-408
# Round 4 = Elite Eight:  501-504
ROUND_RANGES = {
    1: (201, 232),
    2: (301, 316),
    3: (401, 408),
    4: (501, 504),
    5: (601, 602),
    6: (701, 701),
}

rd_results = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

for game in games:
    bracket_pos = game.get('bracketPositionId')
    teams = game.get('teams', [])

    if not teams or len(teams) < 2:
        continue

    # Determine which round this game belongs to
    game_round = None
    for rd, (lo, hi) in ROUND_RANGES.items():
        if lo <= bracket_pos <= hi:
            game_round = rd
            break

    if game_round is None:
        continue

    # isTop=True → Team1 (top of bracket slot), isTop=False → Team2
    team1_data = next((t for t in teams if t.get('isTop') is True), teams[0])
    team2_data = next((t for t in teams if t.get('isTop') is False), teams[1])

    team1 = team1_data['nameShort']
    seed1 = team1_data['seed']
    score1 = team1_data.get('score', 0)

    team2 = team2_data['nameShort']
    seed2 = team2_data['seed']
    score2 = team2_data.get('score', 0)

    winner = team1 if team1_data.get('isWinner', False) else team2
    team1_win = bool(team1_data.get('isWinner', False))

    rd_results[game_round].append(
        [team1, seed1, score1, team2, seed2, score2, winner, team1_win]
    )

for i in range(1, 7):
    print(i)
    for res in rd_results[i]:
        print(res)
    print()

os.makedirs(f'../Data/BracketData/{YEAR}', exist_ok=True)

for i in range(1, 7):
    with open(f'../Data/BracketData/{YEAR}/Round{i}_{YEAR}.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        headings = ['Team1', 'Team1_Seed', 'Team1_Score',
                    'Team2', 'Team2_Seed', 'Team2_Score',
                    'WinningTeam', 'Team1_Win']
        csv_writer.writerow(headings)
        for res in rd_results[i]:
            csv_writer.writerow(res)
