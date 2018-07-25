import datetime

import h5py
import progressbar
import numpy as np
import pandas as pd
import os
import pickle
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from scoring.utils import (Player, get_unique_teams, create_directory,
                            compute_elo_by_game_hdf5_v2, compute_elo_by_goals2,
                            save_players, EloRating)
from pack.utils import delete_directory_contents


class Score(EloRating):

    def __init__(self, training_data=None, champs=None, teams=None,pair_num=None, by='games', temp='./elo_temp/', initial_score=100):
        self.temp = temp
        self.initial_score = initial_score
        self.data = training_data
        self.champs = champs
        self.teams = teams
        self.by = by
        self.pair_num = pair_num

    def main(self):
        matches_champ = self.data[self.data.championship.isin(self.champs)]
        all_teams = get_unique_teams(matches_champ)
        if self.by == 'goals':
            players = [Player(name=p) for p in all_teams]
            players_updated = compute_elo_by_goals2(self.match, matches_champ, players, all_teams)
            save_players(players_updated, self.pair_num, loc=self.temp)
        elif self.by == 'games':
            players = [Player(name=p) for p in all_teams]
            players_updated = compute_elo_by_game_hdf5_v2(self.match, matches_champ, players, all_teams)
            save_players(players_updated, self.pair_num, loc=self.temp)
        else:
            pass


if __name__ == '__main__':

    print('[LOADING]')
    df = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv', index_col='Unnamed: 0')
    df.loc[:, 'date'] = pd.to_datetime(df.date)

    input_date = datetime.datetime(2018, 1, 31)
    input_data = df[df.date == input_date]
    training_data = df.loc[:input_data.index[0]-1]

    #home_teams = input_data.home_team.dropna().drop_duplicates().values
    #teams_grouped = df.groupby('home_team').size()
    #home_teams_consider = teams_grouped[home_teams].sort_values(ascending=False)

    home_teams = input_data.home_team.dropna().drop_duplicates().values
    teams_grouped = df.groupby('home_team').size()
    home_teams_consider = teams_grouped[home_teams].sort_values(ascending=False)

    ### Get Away teams

    away_teams = input_data.away_team.dropna().drop_duplicates().values
    teams_grouped = df.groupby('away_team').size()
    away_teams_consider = teams_grouped[away_teams].sort_values(ascending=False)

    threshold_match_number = 100
    threshold_champ_number = 1000

    home_final_teams = home_teams_consider[home_teams_consider > threshold_match_number].index
    away_final_teams = away_teams_consider[away_teams_consider > threshold_match_number].index

    new_champs = df.groupby('championship').size().sort_values(ascending=False)
    champs = new_champs[new_champs>threshold_champ_number].index.values

    final_matches = input_data[(input_data.home_team.isin(home_final_teams)) & (input_data.away_team.isin(away_final_teams))]
    # final_matches.to_csv('../data/predict.csv')

    pairs = final_matches[['home_team','away_team']].values
    # delete_directory_contents('./elo_temp/')
    # delete_directory_contents(temp)
    try:
        temp = os.environ['EloTempGoals']
    except Exception as e:
        temp = './elo_temp_goals/'
    print(temp)
    try:
        n_start=int(sys.argv[1])
        n_end=int(sys.argv[2])
    except Exception as e:
        print(e)
        n_start=0
        n_end=pairs.shape[0]
    print(n_start,n_end)

    create_directory(temp)
    # delete_directory_contents(temp)

    for p, pair in enumerate(pairs[n_start:n_end, :],start=n_start):
        create_directory(temp)
        print(pair, p + 1, pairs.shape[0])
        champs = df[(df.home_team.isin(pair)) | (df.away_team.isin(pair))].championship.dropna().drop_duplicates().values
        score = Score(champs=champs, pair_num=p, training_data=training_data.iloc[:], by='goals', temp=temp)
        elo_df = score.main()
