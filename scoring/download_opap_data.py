
# coding: utf-8

# In[1]:


from datetime import datetime
import requests
import pandas as pd
import numpy as np
import os, sys

try:
    draw_num = sys.argv[1]
except:
    draw_num = 895

url = "https://pamestoixima.opap.gr/forward/web/services/rs/iFlexBetting/retail/games/15104/0.json?shortTourn=true&startDrawNumber={0}&endDrawNumber={0}&sportId=s-441&marketIds=0&marketIds=0A&marketIds=1&marketIds=69&marketIds=68&marketIds=20&marketIds=21&marketIds=8&locale=en&brandId=defaultBrand&channelId=0".format(str(draw_num))

### sunday

# url = 'https://pamestoixima.opap.gr/forward/web/services/rs/iFlexBetting/retail/games/15104/0.json?shortTourn=true&startDrawNumber=927&endDrawNumber=927&sportId=s-441&marketIds=0&marketIds=0A&marketIds=1&marketIds=69&marketIds=68&marketIds=20&marketIds=21&marketIds=8&locale=en&brandId=defaultBrand&channelId=0'

### GET DATA

s = requests.get(url).json()

datetimes = [datetime.fromtimestamp(d['kdt']/1000) for d in s]

date = datetimes[0].date().__str__()
print(date)
def get_codes(data_json):
    codes = []
    for d in data_json:
        codes.append(int(d['code']))
    return codes

def find_pattern_in_between(s, pat='t-(.+?)_sh'):
    try:
        found = re.search(pat, s).group(1)
        return found
    except AttributeError as e:
        return None              

def get_champs(data_json): 
    champs = []
    for d in data_json[:]:
        key = d['lexicon']['resources'][d['ci']]
        champ = key
        champs.append(champ)
    return champs

def get_home_teams(data_json): 
    home_teams = []
    for d in data_json[:]:
        key = d['lexicon']['resources'][d['hi']]
        home_team = key
        home_teams.append(home_team)
    return home_teams

def get_away_teams(data_json): 
    away_teams = []
    for d in data_json[:]:
        key = d['lexicon']['resources'][d['ai']]
        away_team = key
        away_teams.append(away_team)
    return away_teams

def get_odds(market, ix):
    return [odds['oddPerSet']['1'] for odds in market[ix]['codes']]

def get_all_odds(data_json):
    output = []
    for d in data_json:
        odds = {'FINAL1X2':{'val':None,'key':0,'cols':['home_odds','tie_odds','away_odds']},
            'DC1X2':{'val':None,'key':1,'cols':['1X_odds','12_odds','X2_odds']},
            'HALF1X2':{'val':None,'key':2,'cols':['half_home_odds','half_tie_odds','half_away_odds']},
            'ou25':{'val':None,'key':3,'cols':['over25_odds','under25_odds']},
            'ou35':{'val':None,'key':4,'cols':['over35_odds','under45_odds']},
            'gng':{'val':None,'key':5,'cols':['goal_odds','nogoal_odds']},
            'grange':{'val':[np.nan,np.nan,np.nan,np.nan],'key':6,'cols':['01_odds','23_odds','46_odds','7p_odds']}}
        for key in odds.keys():
            try:
                odd = get_odds(d['markets'], odds[key]['key'])
            except:
                odd = np.nan
            odds[key]['val'] = odd
        output.append(odds)
    return output

codes = get_codes(s)
champs = get_champs(s)
home_teams = get_home_teams(s)
away_teams = get_away_teams(s)
all_odds = get_all_odds(s)

main_df = pd.DataFrame(columns=['code', 'championship', 'home_team', 'away_team'])

main_df.loc[:,'code'] = codes
main_df.loc[:,'championship'] = champs
main_df.loc[:,'home_team'] = home_teams
main_df.loc[:,'away_team'] = away_teams

new_columns = []
odds = all_odds[0]
keys = []
for key in odds.keys():
    # print(key)
    data = odds[key]
    for col in data['cols']:
        new_columns.append(col)
    keys.append(data['key'])

for col in new_columns:
    main_df[col] = np.zeros((main_df.shape[0], 1))

for i, (ix, row) in enumerate(main_df.iloc[:].iterrows()):
    data = all_odds[i]
    for key in data.keys():
        key_data = data[key]
        try:
            for c, d in zip(key_data['cols'],key_data['val']):
                main_df.loc[ix,c] = d
        except:
            continue

basedir='./opap'

try:
    os.mkdir(basedir)
except:
    pass

main_df.head()

main_df.to_csv(basedir+'/'+date+'_opap.csv')

