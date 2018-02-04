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


class Scoring(object):
    """docstring for Scoring."""
    def __init__(self, data=None, teams=None):
        super(Scoring, self).__init__()
        self.teams = np.array(teams) # list
        self.data = data

    def step_1_get_champs_specific_to_teams(self):
        data_of_interest = self.data[['home_team','away_team','championship']]
        champs = data_of_interest[data_of_interest.home_team.isin(self.teams)].championship.dropna().drop_duplicates().values
        del data_of_interest
        return champs

    def step_2_compute_scores_for_each_champ(self):
        champs = self.step_1_get_champs_specific_to_teams()
        elo_df = compute_elo_ratting_dataframe_for_champ_v2(self.data, champs, self.teams)
        try:
            elo_df.to_csv('../data/elo_ratings/ratings/elo.csv')
        except Exception as e:
            elo_df.to_csv('./data/elo_ratings/ratings/elo.csv')
        return elo_df


    def main(self):
        # pass
        self.step_2_compute_scores_for_each_champ()
        # print(df.head())
        # return df



class EloSequencingTestChampionship(object):
    """docstring for EloSequencingTestChampionship."""
    def __init__(self, side_home_or_away='home_team', result_key='over_under_1.5', n_states=2, team_name=None):
        print('[INITIALIZING]')
        super(EloSequencingTestChampionship, self).__init__()
        self.data = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')
        self.champs = pd.read_csv('../champs.csv', nrows=100)
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

    def step1(self):
        print("[RUN] STEP 1 - Computing Score of Teams Per Championship - ")
        chs = range(10)
        for ch in chs[:1]:
            print(self.champs.name[ch])
            team_elo_timeline_df = compute_elo_ratting_dataframe_for_champ(self.data, self.champs, ch)
            print('[SAVING]: ', self.champs.name[ch])
            team_elo_timeline_df.to_csv('../data/elo_ratings/ratings/'+self.champs.name[ch]+'_'+str(ch)+'.csv')

    def step2(self):
        print("[RUN] STEP 2 - Extracting [y] Observable State and [X] Hidden States - ")
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
        print("[RUN] STEP 3 - Computing PX-PXX-PYX for each team")

        files = os.listdir('../data/elo_ratings/states/'+self.side_home_or_away+'/'+self.result_key)
        for f in files[:1]:
            print('[INFO]: CHAMP: ', f)

            bar = progressbar.ProgressBar(widgets=[
                ' [', progressbar.Timer(), '] ',
                progressbar.Bar(),
                ' (', progressbar.ETA(), ') ',
            ])

            try:
                path = '../data/elo_ratings/states/'+self.side_home_or_away+'/'+self.result_key+'/'+f
                champ_name = f.split('_')[0]
                data = pd.read_csv(path)
                data.loc[:, 'x'] = data.x.apply(string_list_to_numpy)
                data.loc[:, 'y'] = data.y.apply(string_list_to_numpy)

                for i in bar(range(data.shape[0])):
                    name = data.loc[i][0].strip('/')
                    # print(name,self.team_name)

                    if self.team_name is not None:
                        if self.team_name != name:
                            continue
                        else:
                            print(self.team_name)

                    x = data.loc[i].x
                    y = data.loc[i].y
                    x_df = pd.DataFrame(x, columns=[self.result_key])
                    y_df = pd.DataFrame(y, columns=[self.result_key])
                    px = x_df[self.result_key].value_counts(normalize=True).sort_index().values
                    pxx = transition_matrix_pxx(data=x_df, result=self.result_key, n_states=self.n_states)
                    pyx = emission_probabilities_pyx(y_df=y_df, x_df=x_df, x_label=self.result_key, hidden_states=self.n_states)
                    probs = {'results': [{'result_key': self.result_key, 'n_states': self.n_states, 'side': self.side_home_or_away,
                                          'px': px, 'pxx': pxx, 'pyx': pyx, 'team_name': name,'x': x, 'y': y,
                                          'championship': champ_name}]}
                    with open('../data/elo_ratings/px_pxx_pyx/'+self.side_home_or_away+'/'+name+'.pickle', 'wb') as handle:
                        pickle.dump(probs, handle)
            except Exception as e:
                print('[Error]', e)

    def step4(self):
        print('[RUN] - STEP 4 - Generating New Sequence - ')
        files = os.listdir('../data/elo_ratings/px_pxx_pyx/'+self.side_home_or_away)
        output = []
        bar = progressbar.ProgressBar(widgets=[
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ])
        for i in bar(range(len(files))[:]):
            file_name = files[i]
            with open('../data/elo_ratings/px_pxx_pyx/'+self.side_home_or_away+'/'+file_name,'rb') as handle:
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

            # try:
            #     fba = ForwardBackwardAlgorithm(px, pxx, pyx, transformed_observed_data_elo_ratings)
            #     forward = fba.forward()
            #
            #     backward = fba.backward()
            #     gammas = fba.gammas()
            # except Exception as e:
            #     continue
            #
            # try:
            #     ac = metrics.accuracy_score(xs, np.argmax(gammas, axis=1))
            #     output.append([team_name,ac, px, pxx, xs.shape[0]])
            # except Exception as e:
            #     pass

        df = pd.DataFrame(output)
        df.columns = ['name','accuracy','px','pxx','n_games']
        df.to_csv('../data/elo_ratings/results/regular/' + self.side_home_or_away + '/' + self.result_key + '/' + 'metrics.csv', index=None)

    def main(self):
        self.directory_setup()
        self.step1()
        # self.step2()
        # self.step3()
        # self.step4()

if __name__ == '__main__':
    data = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')
    teams = ['Yeclano', 'Bnei Yehuda']
    score = Scoring(data = data, teams = teams )
    score.main()
    # runs = [
    #     # ('home_team','result_final', 3),   ('away_team', 'result_final',  3),
    #     ('home_team','over_under_1.5', 2), ('away_team','over_under_1.5', 2),
    #     # ('home_team','over_under_2.5', 2), ('away_team','over_under_2.5', 2),
    #     # ('home_team','over_under_3.5', 2), ('away_team','over_under_3.5', 2),
    #     # ('home_team','over_under_4.5', 2), ('away_team','over_under_4.5', 2)
    #     ]
    #
    # for (side_home_or_away, result_key, n_states) in runs:
    #     print(side_home_or_away, result_key, n_states)
    #     estc = EloSequencingTestChampionship(side_home_or_away=side_home_or_away, result_key=result_key,n_states=n_states, team_name='Acero')
    #     estc.main()
