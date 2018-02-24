import os
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
from trueskill import rate_1vs1

from pack.utils import delete_directory_contents
from scoring.utils import Player


class TrueSkillRating(object):
    def __init__(self, training_data, champs, temp='./trueskill_temp/'):
        self.BETA = ts.BETA
        self.data = training_data
        self.champs = champs
        self.temp = temp
        self.pair_num=pair_num
    def compute_elo_by_game(self, data_df, players, all_teams):
        """
        This function is used to compute the elo ratings of teams based on simply wins and losses.
        :param data_df:
        :param players:
        :param elo:
        :return: array
        """
        n_games = data_df.shape[0]
        elo_table = np.zeros((n_games, len(players)))
        # set initial score for all teams
        elo_table[0, :] = 100
        ix = data_df.index

        for i in range(n_games):
            # print('game: ', i)
            match = data_df.iloc[i]
            player_home = match.home_team
            player_away = match.away_team
            hid = np.where(all_teams == player_home)[0][0]
            aid = np.where(all_teams == player_away)[0][0]
            pair = [players[hid], players[aid]]
            res = match.result_final
            if res == 0:
                a, b = rate_1vs1(pair[0].ranki, pair[1].ranki)

                pair[0].mus.append(a)
                pair[1].mus.append(b)
                pair[0].index.append(ix[i])
                pair[1].index.append(ix[i])

            elif res == 2:
                a, b = rate_1vs1(pair[1].ranki, pair[0].ranki)

                pair[1].mus.append(a)
                pair[0].mus.append(b)
                pair[1].index.append(ix[i])
                pair[0].index.append(ix[i])
            else:
                a, b = rate_1vs1(pair[1].ranki, pair[0].ranki, drawn=True)
                pair[1].mus.append(a)
                pair[0].mus.append(b)
                pair[1].index.append(ix[i])
                pair[0].index.append(ix[i])

        # dfs = []
        # for pl in players:
        #     name = pl.name.replace('/', '')
        #     df = pd.DataFrame(pl.ys, columns=[name], index=pl.index)
        #     df.to_csv(self.temp+'[' + name + '].csv')
        #     dfs.append(df)
        #     del df
        # return pd.concat(dfs, axis=1)

        print('[SAVING]')
        h5f = h5py.File(self.trueskill_temp+'trueskill_pairs_'+str(self.pair_num)+'.h5', 'w')
        pair = 'pair_'+str(self.pair_num)
        h5f.create_group(pair)

        bar = progressbar.ProgressBar(widgets=[
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ])

        for pl in bar(players):
            name = pl.name
            index = pl.index
            ys = pl.mus
            h5f.create_dataset(pair + '/' + name, data=np.array([index, ys]))
        h5f.close()
        return None


    def assemble_dataframes(self):
        print('[RUN]: assembling dataframes')
        files = os.listdir(self.temp)
        dfs = []
        bar = progressbar.ProgressBar(widgets=[
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ])
        for dataframe in bar(files):
            df = pd.read_csv(self.temp+dataframe, index_col='Unnamed: 0')
            df.columns = [dataframe.strip('[].csv')]
            dfs.append(df)
        final = pd.concat(dfs, axis=1, ignore_index=False)
        delete_directory_contents(self.temp)
        return final

    def compute_elo_ratting_dataframe_for_champ_v3(self, data, champs):
        print('[FUNCTION]: Compute Elo for champs')
        matches_champ = data[data.championship.isin(champs)]
        # TODO: dont change the index
        # matches_champ.index = range(matches_champ.shape[0])
        # print('champ', matches_champ.shape)
        # get all the home and away teams
        ht = matches_champ.home_team.values
        at = matches_champ.away_team.values
        # create an array with both home and arway teams regardles of duplicates
        all_teams = np.append(ht, at)
        # drop all the duplicates
        all_teams = np.unique(all_teams)
        # print(all_teams.shape)
        # # create a a Player class for each team. this class stores the elo rating for each player
        players = [Player(name=p) for p in all_teams]
        self.compute_elo_by_game(matches_champ, players, all_teams)
        # ix = np.isin(all_teams, teams)
        # teams_ix = np.where(ix)[0]
        final = self.assemble_dataframes()
        return final

    def win_probability(self, team1, team2):
        delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
        sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
        size = len(team1) + len(team2)
        denom = math.sqrt(size * (self.BETA * self.BETA) + sum_sigma)
        ts = trueskill.global_env()
        return ts.cdf(delta_mu / denom)

    def main(self):
        jsfile = {}
        h5f = h5py.File('trueskill.h5', 'w')

        # try:
        #     delete_directory_contents('../data/elo_ratings/ratings/')
        # except Exception as e:
        #     delete_directory_contents('./data/elo_ratings/ratings/')

        for champ in self.champs[:]:
            h5f.create_group(champ)
            print('[INFO]: ', champ)
            try:
                elo_df = self.compute_elo_ratting_dataframe_for_champ_v3(self.data, [champ]).iloc[:]
                for team in elo_df.columns:
                    ix = elo_df[team].index.values
                    vals = elo_df[team].values.reshape(-1)
                    h5f.create_dataset(champ + '/' + team, data=np.array([ix, vals]))
                jfile = {champ: elo_df}
            except Exception as e:
                print('[ERROR]: Scoring-step_2_1', champ, e)
        print('[SAVING]...')
        h5f.close()

        try:
            with open('../data/elo_ratings/ratings/trueskill.pkl', 'wb') as handle:
                pickle.dump(jsfile, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            with open('./data/elo_ratings/ratings/trueskill.pkl', 'wb') as handle:
                pickle.dump(jsfile, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return jfile


if __name__ == '__main__':

    print('[LOADING]')
    df = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv', index_col='Unnamed: 0')
    df.loc[:, 'date'] = pd.to_datetime(df.date)

    input_date = datetime.datetime(2018, 1, 31)
    input_data = df[df.date == input_date]
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

    score = TrueSkillRating(champs=champs_to_keep, training_data=training_data)
    elo_df = score.main()
