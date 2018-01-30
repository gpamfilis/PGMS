# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing

from utilities import (Elo, Player, transition_matrix_pxx,
                       emission_probabilities_pyx,
                       compute_elo_by_goals, compute_elo_by_game)

from algorithms import ForwardBackwardAlgorithm

print('Reading Champs...')
champs = pd.read_csv('./champs.csv')
print('Reading Data...')
data = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')


chs = range(0, 1)
for ch in chs:
    # get the matches that are from each championship
    matches_champ = data[data.championship == champs.name[ch]]
    print('champ', matches_champ.shape)
    # get all the home and away teams
    ht = matches_champ.home_team.values
    at = matches_champ.away_team.values
    # create an array with both home and arway teams regardles of duplicates
    all_teams = np.append(ht, at)
    # drop all the duplicates
    all_teams = np.unique(all_teams)
    print(all_teams.shape)
    # create a a Player class for each team. this class stores the elo rating for each player
    players = [Player(name=p) for p in all_teams]
    elo = Elo()

    # elo_table_array = compute_elo_by_game(matches_champ, players, all_teams, elo)
    elo_table_array = compute_elo_by_goals(matches_champ, players, all_teams, elo)


# data tech agency
# roisin mcarthy
# alterix

    # team_elo_timeline_df = pd.DataFrame(elo_table_array, columns=all_teams)
    # team_score_tuple = [(p.name, p.score) for p in players]
    # team_score_dataframe = pd.DataFrame(team_score_tuple)
    # team_score_sorted = team_score_dataframe.sort_values(by=[1], ascending=False).iloc[:][0].values
    #
    # output = []
    # for ix_team_name, team_name in enumerate(team_score_sorted[:]):
    #     # state probabilities
    #     px = matches_champ[matches_champ.home_team == team_score_sorted[ix_team_name]].result_final.\
    #         value_counts(normalize=True).sort_index().values
    #     # transition matrix
    #     pxx = transition_matrix_pxx(data=matches_champ[matches_champ.home_team == team_score_sorted[ix_team_name]],
    #                                 result='result_final', n_states=3)
    #
    #     # Emission Probabilities
    #     matches_champ.index = range(matches_champ.shape[0])
    #
    #     observed_data_elo_ratings = team_elo_timeline_df[team_score_sorted[ix_team_name]].diff().replace(np.nan, 0).\
    #         astype(int)[matches_champ[matches_champ.home_team == team_score_sorted[ix_team_name]].index]
    #
    #     unique_observed_data_elo_ratings = np.unique(observed_data_elo_ratings.values)
    #
    #     # emission probabilities
    #     pyx = emission_probabilities_pyx(ys=observed_data_elo_ratings, x_df=matches_champ, x_label='result_final', h_states=3)
    #
    #     # encoding labels
    #     le = preprocessing.LabelEncoder()
    #     le.fit(unique_observed_data_elo_ratings)
    #     transformed_observed_data_elo_ratings = le.transform(observed_data_elo_ratings)
    #
    #     try:
    #         fba = ForwardBackwardAlgorithm(px, pxx, pyx, transformed_observed_data_elo_ratings)
    #         forward = fba.forward()
    #         backward = fba.backward()
    #         gammas = fba.gammas()
    #     except Exception as e:
    #         print(e)
    #         continue
    #     print(ix_team_name, team_name)
    #     try:
    #         ac = metrics.accuracy_score(matches_champ[matches_champ.home_team == team_score_sorted[ix_team_name]].result_final.\
    #                                     values, np.argmax(gammas, axis=1))
    #         output.append([team_name, ac, matches_champ[matches_champ.home_team == team_score_sorted[ix_team_name]].result_final.\
    #                       values.shape[0]])
    #         print('accuracy: ', ac)
    #         # pprint.pprint(ac)
    #     except Exception as e:
    #         print(e)
    #         output.append([team_name, np.nan, np.nan])
    #
    # print('saving')
    # df2 = pd.DataFrame(output)
    # df2.to_csv('./data/elo_ratings/goal_difference/yo_' + str(ch) + '.csv')
