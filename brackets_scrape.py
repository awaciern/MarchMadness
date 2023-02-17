import requests
from bs4 import BeautifulSoup
import re
import csv
import os

URL = 'https://en.wikipedia.org/wiki/2022_NCAA_Division_I_men%27s_basketball_tournament'
ROUND1_FILE_PATH = 'BracketData\\2022\\Round1_2022.csv'
ROUND2_FILE_PATH = 'BracketData\\2022\\Round2_2022.csv'
ROUND3_FILE_PATH = 'BracketData\\2022\\Round3_2022.csv'
ROUND4_FILE_PATH = 'BracketData\\2022\\Round4_2022.csv'
ROUND5_FILE_PATH = 'BracketData\\2022\\Round5_2022.csv'
ROUND6_FILE_PATH = 'BracketData\\2022\\Round6_2022.csv'
FILE_PATH_START = 'BracketData\\2022\\Round'
FILE_PATH_END = '_2022.csv'

headings = ['Team1', 'Team1_Seed', 'Team1_Score',
            'Team2', 'Team2_Seed', 'Team2_Score',
            'WinningTeam', 'Team1_Win']

for i in range(1, 7):
    with open(FILE_PATH_START + str(i) + FILE_PATH_END, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headings)

ROUND_SNAKE = ['1', '2', '1', '3', '1', '2', '1', '4', '1', '2', '1', '3', '1', '2', '1']

page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')

for region in ['West', 'East', 'South', 'Midwest']:
    print()
    print()
    print(region)
    print()
    start = soup.find('span', id=re.compile(region + '_Regional*')).parent
    table = start.find_next_sibling('table')
    table_rows = table.tbody.find_all('tr')

    game_id = 0
    teams_played = 0
    for row in table_rows:
        row_str = str(row)
        winner = False

        nums_unbold = re.findall('>\d+\**\\s*</td>', row_str)
        for i in range(0, len(nums_unbold)):
            nums_unbold[i] = nums_unbold[i].replace('>', '').replace('</td', '').replace('*', '').strip()
        num_bold = re.findall('<b>\d+\**\s*</b>', row_str)
        if num_bold:
            num_bold = num_bold[0].replace('<b>', '').replace('</b>', '').replace('*', '').strip()
        if len(nums_unbold) == 2 and not num_bold:
            seed = nums_unbold[0]
            score = nums_unbold[1]
        elif len(nums_unbold) == 1 and num_bold:
            seed = nums_unbold[0]
            score = num_bold
            winner = True
        else:
            continue

        name = re.findall('>[a-zA-z\'\(\)]+\s*[a-zA-z\'\(\)]*\s*[a-zA-z\'\(\)]*\s*<', row_str)
        # name = re.findall('>(\D|\S)+\s*(\D|\S)*<', row_str)
        # name = re.findall('>\w+(\s*\w+)*<', row_str)
        # print(name)
        if name:
            name = name[0].replace('>', '').replace('<', '').strip()
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
            print(ROUND_SNAKE[game_id])
            print(team1)
            print(team1_seed)
            print(team1_score)
            print(team2)
            print(team2_seed)
            print(team2_score)
            print(winningteam)
            print(team1_win)

            with open(FILE_PATH_START + ROUND_SNAKE[game_id] + FILE_PATH_END, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                row = [team1, team1_seed, team1_score, team2, team2_seed, team2_score, winningteam, team1_win]
                csv_writer.writerow(row)

            teams_played = 0
            game_id += 1

# start = soup.find('span', id=re.compile('Final_Four')).parent
# table = start.find_next_sibling('table')
# table_rows = table.tbody.find_all('tr')
#
# for row in table_rows:
#     game_id = 0
#     teams_played = 0
#     for row in table_rows:
#         row_str = str(row)
#         winner = False
#
#         nums_unbold = re.findall('>\d+\**\\s*</td>', row_str)
#         for i in range(0, len(nums_unbold)):
#             nums_unbold[i] = nums_unbold[i].replace('>', '').replace('</td', '').replace('*', '').strip()
#         num_bold = re.findall('<b>\d+\**\s*</b>', row_str)
#         if num_bold:
#             num_bold = num_bold[0].replace('<b>', '').replace('</b>', '').replace('*', '').strip()
#         if len(nums_unbold) == 2 and not num_bold:
#             seed = nums_unbold[0]
#             score = nums_unbold[1]
#         elif len(nums_unbold) == 1 and num_bold:
#             seed = nums_unbold[0]
#             score = num_bold
#             winner = True
#         else:
#             continue
#
#         name = re.findall('>[a-zA-z\'\(\)]+\s*[a-zA-z\'\(\)]*\s*[a-zA-z\'\(\)]*\s*<', row_str)
#         # name = re.findall('>(\D|\S)+\s*(\D|\S)*<', row_str)
#         # name = re.findall('>\w+(\s*\w+)*<', row_str)
#         # print(name)
#         if name:
#             name = name[0].replace('>', '').replace('<', '').strip()
#         else:
#             print(row_str)
#             print('ERROR: No team name found!')
#             quit()
#
#         if winner:
#             winningteam = name
#         teams_played += 1
#         if teams_played == 1:
#             team1 = name
#             team1_seed = seed
#             team1_score = score
#             team1_win = winner
#         if teams_played == 2:
#             team2 = name
#             team2_seed = seed
#             team2_score = score
#
#             print()
#             print(game_id)
#             print(ROUND_SNAKE[game_id])
#             print(team1)
#             print(team1_seed)
#             print(team1_score)
#             print(team2)
#             print(team2_seed)
#             print(team2_score)
#             print(winningteam)
#             print(team1_win)
#
#             with open(FILE_PATH_START + ROUND_SNAKE[game_id] + FILE_PATH_END, 'a', newline='') as csv_file:
#                 csv_writer = csv.writer(csv_file)
#                 row = [team1, team1_seed, team1_score, team2, team2_seed, team2_score, winningteam, team1_win]
#                 csv_writer.writerow(row)
#
#             teams_played = 0
#             game_id += 1
