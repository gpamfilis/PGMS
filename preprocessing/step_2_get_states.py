# coding: utf-8

# NOTE: this is bullshit why do i have to do this
# TODO: watch david beazley
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pack.utils import (transition_matrix_pxx,
                        emission_probabilities_pyx,
                        compute_elo_ratting_dataframe_for_champ,
                        string_list_to_numpy,
                        delete_directory_contents,
                        compute_elo_ratting_dataframe_for_champ_v2)

from algorithms.hmm import ForwardBackwardAlgorithm

import os
import pickle
import progressbar

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing

class EloSequencingTestChampionship(object):
    """docstring for EloSequencingTestChampionship."""
    def __init__(self, side_home_or_away='home_team', result_key='over_under_1.5', n_states=2,data=None,champs=None, team_name=None):
        print('[INITIALIZING]')
        super(EloSequencingTestChampionship, self).__init__()
        self.data = data #pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')
        self.champs = champs #pd.read_csv('../champs.csv', nrows=100)
        self.result_key = result_key
        self.n_states = n_states
        self.team_name = team_name
        self.side_home_or_away = side_home_or_away
        self.team_name=team_name
        self.teams = None # dataframe with sorted teams
        self.score_function = 'elo'
        self.score_function_name = 'elo'
        self.path_data = None
        self.path_states = '../data/elo_ratings/states/'+self.side_home_or_away
        self.path_states_result_key = self.path_states + '/' + self.result_key
        self.path_px_pxx_py = '../data/elo_ratings/px_pxx_pyx/'+self.side_home_or_away
        self.path_px_pxx_py_side_home_or_away = self.path_px_pxx_py+'/' + self.result_key
        self.results_regular = '../data/elo_ratings/results/regular/' + self.side_home_or_away
        self.results_regular_result_key = self.results_regular + '/' + self.result_key

    def directory_setup(self):

        try:
            os.mkdir(self.path_states)
        except Exception as e:
            delete_directory_contents(self.path_states)
            print('exists', self.result_key, e)

        try:
            os.mkdir(self.path_states_result_key)
        except Exception as e:
            delete_directory_contents(self.path_states_result_key)
            print('exists', self.result_key, e)

        try:
            os.mkdir(self.path_px_pxx_py)
        except Exception as e:
            delete_directory_contents(self.path_px_pxx_py)
            print('exists', self.result_key, e)

        # try:
        #     delete_directory_contents(self.path_px_pxx_py_side_home_or_away)
        # except Exception as e:
        #     print('exists', self.result_key, e)

        try:
            os.mkdir(self.results_regular)
        except:
            print('exists', self.result_key)

        try:
            os.mkdir(self.results_regular_result_key)
        except:
            delete_directory_contents(self.results_regular_result_key)
            print('exists', self.result_key)

    def step21(self):
        print("[RUN] STEP 2 - Extracting [y] Observable State and [X] Hidden States - ")
        data = self.data
        files = os.listdir('../data/elo_ratings/ratings/')
        for f in files[:]:
            print(f)
            championship_name = f.split('.csv')

            bar = progressbar.ProgressBar(widgets=[
                ' [', progressbar.Timer(), '] ',
                progressbar.Bar(),
                ' (', progressbar.ETA(), ') ',
            ])
            elo_ratings_timeline = pd.read_csv('../data/elo_ratings/ratings/'+f, index_col='Unnamed: 0')

            try:
                champ_data = data.loc[elo_ratings_timeline.index]
                output = []
                s_teams = []
                for team_name in bar(elo_ratings_timeline.columns[:]):
                    try:
                        x_data_key = champ_data[self.result_key]
                        observed_data_elo_ratings = elo_ratings_timeline[team_name].diff().replace(np.nan, 0).astype(int)
                        ys = observed_data_elo_ratings.values
                        xs = x_data_key.values
                        out = [champ_data.index.values,champ_data.date.values, ys, xs]
                        output.append(out)
                        s_teams.append(team_name)
                    except Exception as e:
                        print('Error [-] Getting States', team_name, e)

                df = pd.DataFrame(output, columns=['index', 'date', 'y', 'x'], index=s_teams)
                df.to_csv('../data/elo_ratings/states/'+self.side_home_or_away+'/'+self.result_key+'/'+f.strip('.csv') + '.csv')
            except Exception as e:
                print('Error [-] All went to shit', championship_name, self.result_key, e)

    def main(self):
        self.directory_setup()
        self.step21()


if __name__ == '__main__':
    data = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')
    runs = [
        ('home_team', 'result_final',   3), ('away_team', 'result_final',  3),
        # ('home_team','over_under_0.5', 2), ('away_team','over_under_0.5', 2),
        # ('home_team','over_under_1.5', 2), ('away_team','over_under_1.5', 2),
        # ('home_team','over_under_2.5', 2), ('away_team','over_under_2.5', 2),
        # ('home_team','over_under_3.5', 2), ('away_team','over_under_3.5', 2),
        # ('home_team','over_under_4.5', 2), ('away_team','over_under_4.5', 2)
        ]

    for (side_home_or_away, result_key, n_states) in runs:
        print(side_home_or_away, result_key, n_states)
        estc = EloSequencingTestChampionship(side_home_or_away=side_home_or_away, data=data, result_key=result_key, n_states=n_states)
        estc.main()
