# -*- coding: utf-8 -*-

import os
import sys
import pickle

import numpy as np
import pandas as pd

import progressbar


from utilities import (Elo, Player, transition_matrix_pxx,
                       compute_elo_by_goals, compute_elo_by_game)


def string_list_to_numpy(x):
    x = x.strip('[] ').split()
    ar = np.asarray(x).astype(float)
    return ar

def emission_probabilities_pyx(y_df, x_df, x_label='over_under_2.5', hidden_states=2):
    un = np.unique(y_df.values)
    pyx = np.zeros((hidden_states, np.unique(y_df.values).shape[0]))
    # print(un,pyx)
    for i, ix in enumerate(y_df.index):
        try:
            yx = np.where(y_df.loc[ix, x_label] == un)[0][0]
            pyx[int(x_df.loc[ix,x_label]), int(yx)] += 1
        except Exception as e:
            # print('exception',e)
            continue
    for i, s in enumerate(pyx.sum(axis=1)):
        pyx[i, :] = pyx[i, :] / s
    return pyx


try:
    result_key = sys.argv[1]
    n_states = int(sys.argv[2])
    side_home_or_away = sys.argv[3]
except Exception as e:
    print(e)
    print('using default')
    result_key = 'over_under_0.5'
    n_states = 2
    side_home_or_away = 'home_team'
try:
    os.mkdir('./data/elo_ratings/py_pxx_pyx/'+side_home_or_away+'/'+result_key)
except:
    print('exists', result_key)

print('params:',result_key,type(n_states),side_home_or_away)

files = os.listdir('./data/elo_ratings/states/'+side_home_or_away+'/'+result_key)

print('step_2')

for f in files[:1]:
    print(f)

    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])
    try:
        path = './data/elo_ratings/states/'+side_home_or_away+'/'+result_key+'/'+f
        champ_name = f.split('_')[0]
        data = pd.read_csv(path)
        data.loc[:,'x'] = data.x.apply(string_list_to_numpy)
        data.loc[:,'y'] = data.y.apply(string_list_to_numpy)

        for i in bar(range(data.shape[0])):
            name = data.loc[i][0].strip('/')
            x = data.loc[i].x
            y = data.loc[i].y
            x_df = pd.DataFrame(x, columns=[result_key])
            y_df = pd.DataFrame(y, columns=[result_key])
            px = x_df[result_key].value_counts(normalize=True).sort_index().values
            pxx = transition_matrix_pxx(data=x_df, result=result_key, n_states=n_states)
            pyx = emission_probabilities_pyx(y_df=y_df, x_df=x_df, x_label=result_key, hidden_states=3)
            probs = {'results':[{'result_key':result_key,'n_states':n_states, 'side':side_home_or_away,'px':px,'pxx':pxx,'pyx':pyx,'team_name':name,'x':x,'y':y,'championship':champ_name}]}
            with open('./data/elo_ratings/px_pxx_pyx/'+side_home_or_away+'/'+name+'.pickle','wb') as handle:
                pickle.dump(probs, handle)
    except Exception as e:
        print('Error [-]',e)
        pass
