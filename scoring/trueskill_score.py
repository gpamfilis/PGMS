import os
import sys
import math
import pickle
import datetime
import itertools

import numpy as np
import pandas as pd

import h5py
import progressbar

import trueskill
import trueskill as ts
from trueskill import Rating
from trueskill import rate_1vs1

# Personal Packages
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pack.utils import delete_directory_contents
from scoring.utils import Player, compute_trueskill_by_game
from scoring.utils import (Player, get_unique_teams, create_directory,
                            compute_elo_by_game_hdf5_v2, compute_elo_by_goals2,
                            save_players, EloRating, save_players_trueskill)

class TrueSkillRating(object):
    def __init__(self, training_data, champs, pair_num, by,temp='./trueskill_temp/'):
        self.BETA = ts.BETA
        self.data = training_data
        self.champs = champs
        self.temp = temp
        self.by = by
        self.pair_num = pair_num


    def main(self):
        matches_champ = self.data[self.data.championship.isin(self.champs)]
        all_teams = get_unique_teams(matches_champ)
        players = [Player(name=p) for p in all_teams]
        players_updated = compute_trueskill_by_game(matches_champ, players, all_teams)
        save_players_trueskill(players_updated, self.pair_num, loc=self.temp)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_ids_and_mising(f_name='./slavesCombine/trueskill_temp/'):
    files = os.listdir(f_name)
    # https://stackoverflow.com/questions/20718315/how-to-find-a-missing-number-from-a-list
    a = [int(f.split('_')[-1].split('.')[0]) for f in files]
    return list(set(range(a[len(a)-1])[1:]) - set(a))

if __name__ == '__main__':
    try:
        temp = os.environ['TrueSkillTemp']
    except Exception as e:
        temp = './trueskill_temp/'
    print(temp)

    #
    print('[LOADING]')
    df = pd.read_csv('../final_data_soccerway.csv', index_col='Unnamed: 0')
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
        if p not in get_ids_and_mising(temp):
            print('Exists: ',p,pair)
            continue
        else:
            print(pair, p , pairs.shape[0])
            try:
                champs = df[(df.home_team.isin(pair)) | (df.away_team.isin(pair))].championship.dropna().drop_duplicates().values
                score = TrueSkillRating(champs=champs, pair_num=p, training_data=training_data.iloc[:], by='goals', temp=temp)
                elo_df = score.main()
            except Exception as e:
                print(e)
