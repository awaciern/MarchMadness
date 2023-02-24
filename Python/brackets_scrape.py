import requests
from bs4 import BeautifulSoup
import re
import csv
import os

def scrape_bracket_region(region_regex, round_snake):
    print()
    print()
    print(region_regex)
    print()
    start = soup.find('span', id=re.compile(region_regex)).parent
    table = start.find_next_sibling('table')
    table_rows = table.tbody.find_all('tr')

    game_id = 0
    teams_played = 0
    for row in table_rows:
        row_str = str(row)
        winner = False
        name_skip = False

        if region_regex == 'Final_Four*':
            seed = re.findall('[A-Z]+\d+', row_str)
            score_unbold = re.findall('>\d+\**\\s*</td>', row_str)
            score_bold = re.findall('<b>\d+\**\s*</b>', row_str)
            if seed and (score_unbold or score_bold):
                seed = re.findall('\d+', seed[0])[0]
                if score_unbold:
                    score = score_unbold[0].replace('>', '').replace('</td', '').replace('*', '').strip()
                else:
                    score = score_bold[0].replace('<b>', '').replace('</b>', '').replace('*', '').strip()
                    winner = True
            else:
                continue
        else:
            nums_unbold = re.findall('>\d+\**\\s*</td>', row_str)
            for i in range(0, len(nums_unbold)):
                nums_unbold[i] = nums_unbold[i].replace('>', '').replace('</td', '').replace('*', '').strip()
            nums_bold = re.findall('<b>\d+\**\s*</b>', row_str)
            for i in range(0, len(nums_bold)):
                nums_bold[i] = nums_bold[i].replace('<b>', '').replace('</b>', '').replace('*', '').strip()
            if len(nums_unbold) == 2 and not nums_bold:
                seed = nums_unbold[0]
                score = nums_unbold[1]
            elif len(nums_unbold) == 1 and len(nums_bold) == 1:
                seed = nums_unbold[0]
                score = nums_bold[0]
                winner = True
            elif len(nums_bold) == 2 and not nums_unbold:
                seed = nums_bold[0]
                score = nums_bold[1]
                winner = True
            elif YEAR == '2018' and region_regex == 'South_Regional*' and game_id == 0:
                if teams_played == 0:
                    seed = 1
                    score = 54
                    name = 'Virginia'
                    name_skip = True
                else:
                    seed = 16
                    score = 74
                    winner = True
                    name = 'UMBC'
                    name_skip = True
            elif YEAR == '2021' and region_regex == 'West_Regional*' and game_id == 12:
                if teams_played == 0:
                    seed = 7
                    score = None
                    name = 'Oregon'
                    winner = True
                    name_skip = True
                else:
                    seed = 16
                    score = None
                    name = 'VCU'
                    name_skip = True
            else:
                continue

        if not name_skip:
            cg = '[a-zA-z\'\(\)\.&;–]' # regex capture group for team names
            name = re.findall('>' + cg + '+\s*-*'+ cg + '*\s*' + cg + '*\s*<', row_str)
            if name:
                name = name[0].replace('>', '').replace('<', '').replace('&amp;', '&').strip()
            else:
                print(row_str)
                print('ERROR: No team name found!')
                quit()

        if winner:
            winningteam = name
        teams_played += 1
        if teams_played == 1:
            team1 = name
            team1_seed = seed
            team1_score = score
            team1_win = winner
        if teams_played == 2:
            team2 = name
            team2_seed = seed
            team2_score = score

            print()
            print(game_id)
            print(round_snake[game_id])
            print(team1)
            print(team1_seed)
            print(team1_score)
            print(team2)
            print(team2_seed)
            print(team2_score)
            print(winningteam)
            print(team1_win)

            with open(FILE_PATH_START + round_snake[game_id] + FILE_PATH_END, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                row = [team1, team1_seed, team1_score, team2, team2_seed, team2_score, winningteam, team1_win]
                csv_writer.writerow(row)

            teams_played = 0
            game_id += 1

YEAR = '2021'
URL = 'https://en.wikipedia.org/wiki/' + YEAR + '_NCAA_Division_I_men%27s_basketball_tournament'
ROUND1_FILE_PATH = '..\\Data\\BracketData\\'+ YEAR +'\\Round1_' + YEAR + '.csv'
ROUND2_FILE_PATH = '..\\Data\\BracketData\\'+ YEAR +'\\Round2_' + YEAR + '.csv'
ROUND3_FILE_PATH = '..\\Data\\BracketData\\'+ YEAR +'\\Round3_' + YEAR + '.csv'
ROUND4_FILE_PATH = '..\\Data\\BracketData\\'+ YEAR +'\\Round4_' + YEAR + '.csv'
ROUND3_FILE_PATH = '..\\Data\\BracketData\\'+ YEAR +'\\Round6_' + YEAR + '.csv'
ROUND5_FILE_PATH = '..\\Data\\BracketData\\'+ YEAR +'\\Round5_' + YEAR + '.csv'
FILE_PATH_START = '..\\Data\\BracketData\\'+ YEAR +'\\Round'
FILE_PATH_END = '_' + YEAR + '.csv'

headings = ['Team1', 'Team1_Seed', 'Team1_Score',
            'Team2', 'Team2_Seed', 'Team2_Score',
            'WinningTeam', 'Team1_Win']

if not os.path.exists('..\\BracketData\\'+ YEAR):
    os.mkdir('..\\BracketData\\'+ YEAR)

for i in range(1, 7):
    with open(FILE_PATH_START + str(i) + FILE_PATH_END, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headings)

page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')

round_snake_regional = ['1', '2', '1', '3', '1', '2', '1', '4', '1', '2', '1', '3', '1', '2', '1']
for region in ['West', 'East', 'South', 'Midwest']:
    region_regex = region + '_Regional*'
    scrape_bracket_region(region_regex, round_snake_regional)

round_snake_finalfour = ['5', '6', '5']
scrape_bracket_region('Final_Four*', round_snake_finalfour)
