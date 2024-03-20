import requests
from bs4 import BeautifulSoup
import re
import csv
import pandas as pd

# URL = 'https://web.archive.org/web/20230313204645/https://kenpom.com/'
URL = 'https://web.archive.org/web/20240318160959/https://kenpom.com/'
# FILE_PATH = '..\\Data\\KenPomData\\2023_0.csv'
FILE_PATH = '../Data/KenPomData/2024.csv'
PYTHAGOREAN = False

page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
data_area = soup.find(id='data-area')
table = data_area.find('table')

headings = ['Rk_AdjEM', 'Team', 'Seed', 'Conf',
            'W-L', 'Wins', 'Losses', 'WinPct', 'AdjEM',
            'AdjO', 'Rk_AdjO', 'AdjD', 'Rk_AdjD', 'AdjT', 'Rk_AdjT',
            'Luck', 'Rk_Luck', 'SOS_AdjEM', 'Rk_SOS_AdjEM', 'SOS_AdjO',
            'Rk_SOS_AdjO', 'SOS_AdjD', 'Rk_SOS_AdjD',
            'NCSOS_AdjEM', 'Rk_NCSOS_AdjEM']
# print(headings)
# table_headings = table.tbody.find_all("th")
# headings = []
# for heading in table_headings:
#     text = heading.text
#     if text != "" and text not in headings:
#         headings.append(text)

table_data = table.tbody.find_all('tr', class_='tourney')
data = []
for team in table_data:
    stats = []
    for i, stat in enumerate(team.find_all('td')):
        stat = stat.text
        if i == 1:
            seed = re.search('\d+', stat)
            name = re.sub('\s\d+', '', stat)
            stats.append(name)
            stats.append(int(seed.group(0)))
        elif i == 2:
            stats.append(stat)
        elif i == 3:
            stats.append(stat)
            win_loss_list = re.findall('\d+', stat)
            wins = int(win_loss_list[0])
            losses = int(win_loss_list[1])
            stats.append(wins)
            stats.append(losses)
            stats.append(wins / (wins + losses))
        elif PYTHAGOREAN and i in [4, 13, 19]:
            stats.append(None)
        elif i == 0 or (i > 5 and i % 2 == 0):
            stats.append(int(stat))
        else:
            stats.append(float(stat))
    if PYTHAGOREAN:
        stats[8] = round(stats[9] - stats[11], 2)
        stats[17] = round(stats[19] - stats[21], 2)
    data.append(stats)

with open(FILE_PATH, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(headings)
    for row in data:
        csv_writer.writerow(row)

if PYTHAGOREAN:
    df = pd.read_csv(FILE_PATH)
    df = df.sort_values(by='SOS_AdjEM', ascending=False)
    df['Rk_SOS_AdjEM'] = [i for i in range(1, len(df)+1)]
    df = df.sort_values(by='AdjEM', ascending=False)
    df['Rk_AdjEM'] = [i for i in range(1, len(df)+1)]
    df.to_csv(path_or_buf=FILE_PATH, index=False)
