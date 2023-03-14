import pandas as pd

'''
NOTE: The following team name changes were manually made to bracket data:
Connecticut -> UConn
North Carolina State -> NC State
SMU -> Southern Methodist
UC Irvine -> California-Irvine
Virginia Commonwealth -> VCU

NOTE: The following teams were mismatched in KenPom only due to being in play-in games (no reconcile needed):
Boise State (2013 - as a 13 seed, WIERD :P)
Long Island (2013)
Providence (2017)
'''

kp_to_game_name_dict = {}
kp_to_game_name_dict['TODO'] = 'TODO'
kp_to_game_name_dict['Arizona St.'] = 'Arizona State'
kp_to_game_name_dict['Arkansas Little Rock'] = 'Little Rock'
kp_to_game_name_dict['Boise St.'] = 'Boise State'
kp_to_game_name_dict['Brigham Young'] = 'BYU'
kp_to_game_name_dict['Cal St. Bakersfield'] = 'Cal State Bakersfield'
kp_to_game_name_dict['Cal St. Fullerton'] = 'Cal State Fullerton'
kp_to_game_name_dict['Cleveland St.'] = 'Cleveland State'
kp_to_game_name_dict['Colorado St.'] = 'Colorado State'
kp_to_game_name_dict['Connecticut'] = 'UConn'
kp_to_game_name_dict['East Tennessee St.'] = 'East Tennessee State'
kp_to_game_name_dict['Florida St.'] = 'Florida State'
kp_to_game_name_dict['Fresno St.'] = 'Fresno State'
kp_to_game_name_dict['Gardner Webb'] = 'Gardner-Webb'
kp_to_game_name_dict['Georgia St.'] = 'Georgia State'
kp_to_game_name_dict['Iowa St.'] = 'Iowa State'
kp_to_game_name_dict['Jacksonville St.'] = 'Jacksonville State'
kp_to_game_name_dict['Kansas St.'] = 'Kansas State'
kp_to_game_name_dict['Kennesaw St.'] = 'Kennesaw State'
kp_to_game_name_dict['Kent St.'] = 'Kent State'
kp_to_game_name_dict['Long Beach St.'] = 'Long Beach State'
kp_to_game_name_dict['Louisiana Lafayette'] = 'Louisiana-Lafayette'
kp_to_game_name_dict['Loyola MD'] = 'Loyola (MD)'
kp_to_game_name_dict['Miami FL'] = 'Miami (FL)'
kp_to_game_name_dict['Michigan St.'] = 'Michigan State'
kp_to_game_name_dict['Mississippi'] = 'Ole Miss'
kp_to_game_name_dict['Mississippi St.'] = 'Mississippi State'
kp_to_game_name_dict['Montana St.'] = 'Montana State'
kp_to_game_name_dict['Morehead St.'] = 'Morehead State'
kp_to_game_name_dict['Murray St.'] = 'Murray State'
kp_to_game_name_dict['NC Asheville'] = 'UNC Asheville'
kp_to_game_name_dict['Nevada Las Vegas'] = 'UNLV'
kp_to_game_name_dict['New Mexico St.'] = 'New Mexico State'
kp_to_game_name_dict['Norfolk St.'] = 'Norfolk State'
kp_to_game_name_dict['North Carolina Central'] = 'NC Central'
kp_to_game_name_dict['North Carolina St.'] = 'NC State'
kp_to_game_name_dict['N.C. State'] = 'NC State'
kp_to_game_name_dict['North Dakota St.'] = 'North Dakota State'
kp_to_game_name_dict['Northwestern St.'] = 'Northwestern State'
kp_to_game_name_dict['Ohio St.'] = 'Ohio State'
kp_to_game_name_dict['Oklahoma St.'] = 'Oklahoma State'
kp_to_game_name_dict['Oregon St.'] = 'Oregon State'
kp_to_game_name_dict['Penn St.'] = 'Penn State'
kp_to_game_name_dict['SMU'] = 'Southern Methodist'
kp_to_game_name_dict['San Diego St.'] = 'San Diego State'
kp_to_game_name_dict['South Dakota St.'] = 'South Dakota State'
kp_to_game_name_dict['Southern Mississippi'] = 'Southern Miss'
kp_to_game_name_dict['St. Louis'] = 'Saint Louis'
kp_to_game_name_dict['St. Mary\'s'] = 'Saint Mary\'s'
kp_to_game_name_dict['Texas A&M Corpus Chris'] = 'Texas A&M–CC'
kp_to_game_name_dict['UC Irvine'] = 'California-Irvine'
kp_to_game_name_dict['Utah St.'] = 'Utah State'
kp_to_game_name_dict['Virginia Commonwealth'] = 'VCU'
kp_to_game_name_dict['Weber St.'] = 'Weber State'
kp_to_game_name_dict['Wichita St.'] = 'Wichita State'
kp_to_game_name_dict['Wright St.'] = 'Wright State'

for year in range(2023, 2024):
    if year == 2020:
        continue

    FILE_PATH = '..\\Data\\KenPomData\\' + str(year) + '.csv'
    df_kp = pd.read_csv(FILE_PATH)

    for key in kp_to_game_name_dict:
        df_kp.loc[df_kp['Team'] == key, 'Team'] = kp_to_game_name_dict[key]

    df_kp.to_csv(path_or_buf=FILE_PATH, index=False)
