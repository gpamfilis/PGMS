import datetime
import pandas as pd
import numpy as np
import pickle

from .utils import (compute_elo_ratting_dataframe_for_champ_v2, compute_elo_ratting_dataframe_for_champ_v3, delete_directory_contents)
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

class Scoring(object):
    """docstring for Scoring."""
    def __init__(self, data=None, champs=None, teams=None):
        super(Scoring, self).__init__()
        self.teams = np.array(teams)  # list
        self.data = data
        self.champs = champs

    def step_1_get_champs_specific_to_teams(self):
        data_of_interest = self.data[['home_team', 'away_team','championship']]
        champs = data_of_interest[data_of_interest.home_team.isin(self.teams)].championship.dropna().drop_duplicates().values
        del data_of_interest
        return champs

    def step_2_compute_scores_for_each_champ(self):
        champs = self.step_1_get_champs_specific_to_teams()
        elo_df = compute_elo_ratting_dataframe_for_champ_v2(self.data, champs)
        try:
            elo_df.to_csv('../data/elo_ratings/ratings/elo.csv')
        except Exception as e:
            elo_df.to_csv('./data/elo_ratings/ratings/elo.csv')
        return elo_df

    def step_2_1(self):
        jfile = {}

        try:
            delete_directory_contents('../data/elo_ratings/ratings/')
        except Exception as e:
            delete_directory_contents('./data/elo_ratings/ratings/')

        for champ in self.champs[:1]:
            print('[INFO]: ', champ)

            try:
                elo_df = compute_elo_ratting_dataframe_for_champ_v3(self.data, [champ]).iloc[:10]
                jfile = {champ:elo_df}
            except Exception as e:
                print('[ERROR]: Scoring-step_2_1', champ, e)
        return jfile


    def savit(self):
        print('[SAVING]...')
        try:
            with open('../data/elo_ratings/ratings/data.pkl', 'wb') as handle:
                pickle.dump(jsfile, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            with open('./data/elo_ratings/ratings/data.pkl', 'wb') as handle:
                pickle.dump(jsfile, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def main(self):
        if self.champs is not None:
            self.step_2_1()

if __name__ == '__main__':
    df = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv', index_col='Unnamed: 0')
    df.loc[:,'date'] = pd.to_datetime(df.date)

    input_date = datetime.datetime(2018,1,31)
    input_data = df[df.date==input_date]
    training_data = df.loc[:input_data.index[0]-1]

    home_teams = input_data.home_team.dropna().drop_duplicates().values
    teams_grouped = df.groupby('home_team').size()
    home_teams_consider = teams_grouped[home_teams].sort_values(ascending=False)

    away_teams = input_data.away_team.dropna().drop_duplicates().values
    teams_grouped = df.groupby('away_team').size()
    away_teams_consider = teams_grouped[away_teams].sort_values(ascending=False)

    threshold = 300
    home_final_teams = home_teams_consider[home_teams_consider > threshold].index
    away_final_teams = away_teams_consider[away_teams_consider > threshold].index

    final_matches = input_data[(input_data.home_team.isin(home_final_teams)) & (input_data.away_team.isin(away_final_teams))]
    final_matches.to_csv('../data/predict.csv')
    champs_to_keep = final_matches.championship.dropna().drop_duplicates().values

    score = Scoring(data = training_data, champs= champs_to_keep, teams=None)
    elo_df = score.main()
