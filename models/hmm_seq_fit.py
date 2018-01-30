# coding: utf-8

from pack.utils import (transition_matrix_pxx,
                        emission_probabilities_pyx,
                        compute_elo_ratting_dataframe_for_champ,
                        string_list_to_numpy)
from algorithms.algorithms import ForwardBackwardAlgorithm

import os
import pickle

import numpy as np
import pandas as pd

import progressbar

# NOTE: this is bullshit why do i have to do this
# TODO: watch david beazley
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


class EloSequencingTestChampionship(object):
    """docstring for EloSequencingTestChampionship."""
    def __init__(self):
        super(EloSequencingTestChampionship, self).__init__()
        self.data = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')
        self.champs = pd.read_csv('../champs.csv',nrows=100)
        self.result_key = 'over_under_1.5'
        self.n_states = 2
        self.side_home_or_away = 'home_team'
        self.team_name=None
        self.teams = None # dataframe with sorted teams
        self.score_function = 'elo'
        self.score_function_name = 'elo'
        self.path_data = None

    def directory_setup(self):

        try:
            os.mkdir('../data/elo_ratings/states/'+self.side_home_or_away)
        except Exception as e:
            print('exists', self.result_key, e)

        try:
            os.mkdir('../data/elo_ratings/states/'+self.side_home_or_away+'/'+self.result_key)
        except Exception as e:
            print('exists', self.result_key, e)

        try:
            os.mkdir('../data/elo_ratings/px_pxx_pyx/'+self.side_home_or_away)
        except Exception as e:
            print('exists', self.result_key, e)

        try:
            os.mkdir('../data/elo_ratings/py_pxx_pyx/'+self.side_home_or_away+'/'+self.result_key)
        except Exception as e:
            print('exists', self.result_key, e)

    def step1(self):
        print("[RUN] STEP 1 - Computing Score of Teams Per Championship - ")
        chs = range(10)
        for ch in chs[:1]:
            print(self.champs.name[ch])
            team_elo_timeline_df = compute_elo_ratting_dataframe_for_champ(self.data, self.champs, ch)
            print('[SAVING]: ', self.champs.name[ch] )
            team_elo_timeline_df.to_csv('../data/elo_ratings/ratings/'+self.champs.name[ch]+'_'+str(ch)+'.csv',
                                        index=None)

    def step2(self):
        print("[RUN] STEP 1 - Computing Score of Teams Per Championship - ")

        data = self.data
        files = os.listdir('../data/elo_ratings/ratings/')
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
                elo_ratings_timeline = pd.read_csv('../data/elo_ratings/ratings/'+f)
                champ_data.index = range(champ_data.shape[0])
                output = []
                s_teams = []
                for team_name in bar(elo_ratings_timeline.columns[:]):
                    try:
                        x_data = champ_data[champ_data[self.side_home_or_away] == team_name]
                        x_data_key = x_data[self.result_key]
                        observed_data_elo_ratings = elo_ratings_timeline[team_name].diff().replace(np.nan, 0).astype(int)
                        ys = observed_data_elo_ratings[x_data.index]
                        output.append([ys.values, x_data_key.values])
                        s_teams.append(team_name)
                    except Exception as e:
                        print('Error [-] Getting States', team_name, e)

                df = pd.DataFrame(output, columns=['y', 'x'], index=s_teams)
                df.to_csv('../data/elo_ratings/states/'+self.side_home_or_away+'/'+self.result_key+'/'+championship_name
                          + '_' + '[' + self.result_key + ']' + '_' + str(self.n_states) + '_' + 'y_state_x_state.csv')
            except Exception as e:
                print('Error [-] All went to shit', championship_name, self.result_key,e)


    def step3(self):

        result_key = self.result_key
        n_states = self.n_states
        side_home_or_away = self.side_home_or_away

        files = os.listdir('../data/elo_ratings/states/'+side_home_or_away+'/'+result_key)

        print('step_2')

        for f in files[:1]:
            print(f)

            bar = progressbar.ProgressBar(widgets=[
                ' [', progressbar.Timer(), '] ',
                progressbar.Bar(),
                ' (', progressbar.ETA(), ') ',
            ])
            try:
                path = '../data/elo_ratings/states/'+side_home_or_away+'/'+result_key+'/'+f
                champ_name = f.split('_')[0]
                data = pd.read_csv(path)
                data.loc[:, 'x'] = data.x.apply(string_list_to_numpy)
                data.loc[:, 'y'] = data.y.apply(string_list_to_numpy)

                for i in bar(range(data.shape[0])):
                    name = data.loc[i][0].strip('/')
                    x = data.loc[i].x
                    y = data.loc[i].y
                    x_df = pd.DataFrame(x, columns=[result_key])
                    y_df = pd.DataFrame(y, columns=[result_key])
                    px = x_df[result_key].value_counts(normalize=True).sort_index().values
                    pxx = transition_matrix_pxx(data=x_df, result=result_key, n_states=n_states)
                    pyx = emission_probabilities_pyx(y_df=y_df, x_df=x_df, x_label=result_key, hidden_states=3)
                    probs = {'results': [{'result_key': result_key, 'n_states': n_states, 'side': side_home_or_away,
                                          'px': px, 'pxx': pxx, 'pyx': pyx, 'team_name': name,'x': x, 'y': y,
                                          'championship': champ_name}]}
                    with open('../data/elo_ratings/px_pxx_pyx/'+side_home_or_away+'/'+name+'.pickle', 'wb') as handle:
                        pickle.dump(probs, handle)
            except Exception as e:
                print('[Error]', e)

    def step4(self):
        pass

    def main(self):
        self.directory_setup()
        # self.step1()
        self.step2()
        self.step3()
        self.step4()


if __name__ == '__main__':
    estc = EloSequencingTestChampionship()
    estc.main()
