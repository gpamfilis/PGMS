import os
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn import preprocessing

# https://stackoverflow.com/questions/38601026/easy-way-to-use-parallel-options-of-scikit-learn-functions-on-hpc/38814491#38814491

# LOADING DATA
print('[LOADING...]')
df = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv', index_col='Unnamed: 0')
pred_data = pd.read_csv('../data/predict.csv', index_col='Unnamed: 0')
final = pd.read_csv('./final.csv', index_col='Unnamed: 0')

# GET UNIQUE TEAMS
print('[GETTING UNIQUE TEAMS]')
ht = final.home_team.values
at = final.away_team.values
all_teams = np.append(ht,at)
un = np.unique(all_teams)

# ENCODE THE TEAM NAMES
print('[ENCODE THE TEAM NAMES]')
le = preprocessing.LabelEncoder()
le.fit(un)
final.loc[:,'home_team'] = le.transform(final.home_team.values)
final.loc[:,'away_team'] = le.transform(final.away_team.values)

# GET RESULTS OF CHOICE
print('[GET RESULTS OF CHOICE]')
res_index = final.res_index.values
out = df.loc[res_index]
result_key = 'result_final'
final.loc[:, result_key] = out[result_key].values

# FINAL WRAPUP
print('[FINAL WRAPUP]')
final = final.dropna()
final2 = final.drop('res_index', axis=1)
final2.to_csv('./final_trans.csv')
