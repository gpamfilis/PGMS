# -*- coding: utf-8 -*-

import os
import sys
import pickle

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing

import progressbar

from algorithms import ForwardBackwardAlgorithm

try:
    result_key = sys.argv[1]
    side_home_or_away = sys.argv[2]
except Exception as e:
    print(e)
    print('using default')
    result_key = 'over_under_0.5'
    side_home_or_away = 'home_team'
try:
    os.mkdir('./data/elo_ratings/results/regular/'+side_home_or_away)
except:
    print('exists', result_key)

try:
    os.mkdir('./data/elo_ratings/results/regular/'+side_home_or_away+'/'+result_key)
except:
    print('exists', result_key)

print('params:',side_home_or_away,result_key)

files = os.listdir('./data/elo_ratings/px_pxx_pyx/'+side_home_or_away)
output = []
bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
])
print('step_3')
for i in bar(range(len(files))[:]):
    file_name = files[i]
    with open('./data/elo_ratings/px_pxx_pyx/'+side_home_or_away+'/'+file_name,'rb') as handle:
        probs = pickle.load(handle)

    px = probs['results'][0]['px']
    pxx = probs['results'][0]['pxx']
    pyx = probs['results'][0]['pyx']
    ys = probs['results'][0]['y']
    xs = probs['results'][0]['x']
    result_key = probs['results'][0]['result_key']
    unique_observed_data_elo_ratings = np.unique(ys)

    le = preprocessing.LabelEncoder()
    le.fit(unique_observed_data_elo_ratings)
    transformed_observed_data_elo_ratings = le.transform(ys)

    team_name=probs['results'][0]['team_name']

    try:
        fba = ForwardBackwardAlgorithm(px, pxx, pyx, transformed_observed_data_elo_ratings)
        forward = fba.forward()
        backward = fba.backward()
        gammas = fba.gammas()
    except Exception as e:
        continue

    try:
        ac = metrics.accuracy_score(xs, np.argmax(gammas, axis=1))
        output.append([team_name,ac,  xs.shape[0]])
    except Exception as e:
        pass

df = pd.DataFrame(output)
df.columns = ['name','accuracy','n_games']
df.to_csv('./data/elo_ratings/results/regular/'+side_home_or_away+'/'+result_key+'/'+'metrics.csv', index=None)
