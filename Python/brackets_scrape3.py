import requests
from bs4 import BeautifulSoup
import re
import csv
import os

YEAR = '2024'
URL = 'https://en.wikipedia.org/wiki/' + YEAR + '_NCAA_Division_I_men%27s_basketball_tournament'

page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')

# print(soup)

region_regex = 'East_regional*'

rd_matchup_idxs = {
    1 : [
        [0, 1],
        [4, 5],
        [8, 9],
        [12, 13],
        [16, 17],
        [20, 21],
        [24, 25],
        [28, 29]
    ],
    2: [
        [2, 3],
        [10, 11],
        [18, 19],
        [26, 27],
    ],
    3: [
        [6, 7],
        [22, 23],
    ], 
    4: [
        [14, 15],
    ]
}

rd_results = {
    1: [],
    2: [],
    3: [], 
    4: []
}

for region_regex in ['East_regional*', 'West_regional*', 'South_regional*', 'Midwest_regional*']:
    team_result_list = []
    start = soup.find('span', id=re.compile(region_regex)).parent.parent
    # print(start)
    table = start.find_next_sibling('table')
    # print(table)
    table_rows = table.tbody.find_all('tr')
    for i, row in enumerate(table_rows):
        row_str = str(row)
        # print(row_str)
        # nums_re = re.findall(r'>\d+(?:<sup>OT<\/sup>)?\s<\/td><td rowspan=', row_str)
        nums_re = re.findall(r'>\d+(?:<sup>\d*OT<\/sup>)?\s<\/td><td ', row_str)
        # print(nums_re)
        if "South Dakota State" in row_str:
            print(row_str)
            print(nums_re)
            print()
        if len(nums_re) != 2:
            continue
        seed = int(re.findall(r'\d+', nums_re[0])[0])
        # print(seed)
        score = int(re.findall(r'\d+', nums_re[1])[0].replace('<sup>OT</sup>', ''))
        # print(score)
        name_re = re.findall(r'>[^0-9]+\s+<\/td><td rowspan=', row_str)
        name = name_re[0][1:name_re[0].index('</td>')].replace('</a>', '').strip()
        # print(name)
        # print(name_re)
        # if i == 5:
        #     break
        # print()
        team_result_list.append([name, seed, score])

    for i, result in enumerate(team_result_list):
        print(f'{i}: {result}')
    print()

    for i in range(1, 5):
        for  matchup_idxs in rd_matchup_idxs[i]:
            team1 = team_result_list[matchup_idxs[0]][0]
            seed1 = team_result_list[matchup_idxs[0]][1]
            score1 = team_result_list[matchup_idxs[0]][2]
            team2 = team_result_list[matchup_idxs[1]][0]
            seed2 = team_result_list[matchup_idxs[1]][1]
            score2 = team_result_list[matchup_idxs[1]][2]
            rd_results[i].append([team1, seed1, score1, team2, seed1, score2, team1 if score1 > score2 else team2, score1 > score2])

for i in range(1, 5):
    print(i)
    for res in rd_results[i]:
        print(res) 
    print()

for i in range(1, 5):
    with open(f'../Data/BracketData/2024/Round{i}_2024.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        headings = ['Team1', 'Team1_Seed', 'Team1_Score',
            'Team2', 'Team2_Seed', 'Team2_Score',
            'WinningTeam', 'Team1_Win']
        csv_writer.writerow(headings)
        for res in rd_results[i]:
            csv_writer.writerow(res)

# DO FINAL FOUR MANUALLY!
