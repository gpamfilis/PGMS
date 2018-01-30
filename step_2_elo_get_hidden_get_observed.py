import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utilities import (Elo, Player, transition_matrix_pxx,
                       emission_probabilities_pyx,
                       compute_elo_by_goals, compute_elo_by_game)
import os
import sys
import progressbar


data = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')

# champs = pd.read_csv('./champs.csv',nrows=10)

try:
    result_key = sys.argv[1]
    n_states = sys.argv[2]
    side_home_or_away = sys.argv[3]
except Exception as e:
    print(e)
    result_key = 'over_under_1.5'
    n_states = 2
    side_home_or_away = 'home_team'

try:
    os.mkdir('./data/elo_ratings/states/'+side_home_or_away)
except:
    print('exists',result_key)

try:
    os.mkdir('./data/elo_ratings/states/'+side_home_or_away+'/'+result_key)
except:
    print('exists',result_key)

files = os.listdir('./data/elo_ratings/ratings/')
print('step_2')
for f in files[:1]:
    print(f)
    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])

    try:
        championship_name = f.split('_')[0]
        champ_data = data[data.championship==championship_name]
        elo_ratings_timeline = pd.read_csv('./data/elo_ratings/ratings/'+f)
        champ_data.index = range(champ_data.shape[0])
        output = []
        s_teams = []
        for team_name in bar(elo_ratings_timeline.columns[:]):
            try:
                x_data = champ_data[champ_data[side_home_or_away] == team_name]
                x_data_key = x_data[result_key]
                observed_data_elo_ratings = elo_ratings_timeline[team_name].diff().replace(np.nan, 0).astype(int)
                ys = observed_data_elo_ratings[x_data.index]
                output.append([ys.values, x_data_key.values])
                s_teams.append(team_name)
            except Except as e:
                print('Error [-] Getting States',team_name,e)
                pass

        df = pd.DataFrame(output,columns=['y','x'],index=s_teams)
        df.to_csv('./data/elo_ratings/states/'+side_home_or_away+'/'+result_key+'/'+championship_name+'_'+'['+result_key+']'+'_'+str(n_states)+'_'+'y_state_x_state.csv')
    except Exception as e:
        print('Error [-] All went to shit',championship_name,result_key,e)
        pass
