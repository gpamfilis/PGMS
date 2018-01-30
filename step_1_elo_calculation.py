# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing

from utilities import (Elo, Player, transition_matrix_pxx,
                       emission_probabilities_pyx,
                       compute_elo_by_goals, compute_elo_by_game
                       )

from algorithms import ForwardBackwardAlgorithm




def compute_elo_ratting_dataframe_for_champ(data,champs, champ_ix):
    matches_champ = data[data.championship == champs.name[champ_ix]]
    matches_champ.index = range(matches_champ.shape[0])
    print('champ', matches_champ.shape)
    # get all the home and away teams
    ht = matches_champ.home_team.values
    at = matches_champ.away_team.values
    # create an array with both home and arway teams regardles of duplicates
    all_teams = np.append(ht, at)
    # drop all the duplicates
    print(all_teams.shape)
    all_teams = np.unique(all_teams)
    # create a a Player class for each team. this class stores the elo rating for each player
    players = [Player(name=p) for p in all_teams]
    elo = Elo()

    # elo_table_array = compute_elo_by_game(matches_champ, players, all_teams, elo)
    elo_table_array = compute_elo_by_goals(matches_champ, players, all_teams, elo)

    team_elo_timeline_df = pd.DataFrame(elo_table_array, columns=all_teams)
    return team_elo_timeline_df


if __name__ == '__main__':
    print('Reading Champs...')
    champs = pd.read_csv('./champs.csv',nrows=100)
    print('Reading Data...')
    data = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')
    chs = range(10)
    for ch in chs[:1]:
        print(champs.name[ch])
        team_elo_timeline_df = compute_elo_ratting_dataframe_for_champ(data, champs, ch)
        team_elo_timeline_df.to_csv('./data/elo_ratings/ratings/'+champs.name[ch]+'_'+str(ch)+'.csv',index=None)
