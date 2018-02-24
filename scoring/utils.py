from trueskill import Rating
import os
import progressbar
import pandas as pd
import numpy as np
import h5py


def create_directory(dir_name):
    print('[FUNCTION]: checking if ',dir_name,' exists.')
    try:
        os.mkdir(dir_name)
    except Exception as e:
        pass
    return None




class EloRating(object):

    def match(self, p1, p2):
        return self.match_algo_strict2(p1, p2)

    @staticmethod
    def match_algo_strict(previous_winner_score, previous_looser_score, K=10):
        r1 = max(min(previous_looser_score - previous_winner_score, 400), -400)
        r2 = max(min(previous_winner_score - previous_looser_score, 400), -400)
        e1 = 1.0 / (1+10**(r1 / 400))
        e2 = 1.0 / (1+10**(r2 / 400))
        s1 = 1
        s2 = 0
        new_winner_score = previous_winner_score + K*(s1-e1)
        new_looser_score = previous_looser_score + K*(s2-e2)

        return new_winner_score, new_looser_score

    @staticmethod
    def match_algo_strict2(winner, looser, K=10):
        r1 = max(min(looser.score - winner.score, 400), -400)
        r2 = max(min(winner.score - looser.score, 400), -400)
        e1 = 1.0 / (1+10**(r1 / 400))
        e2 = 1.0 / (1+10**(r2 / 400))
        s1 = 1
        s2 = 0
        winner.score = winner.score + K*(s1-e1)
        looser.score = looser.score + K*(s2-e2)

        # increase win counter
        winner.wins += 1

        # increase match counter
        winner.matches += 1
        looser.matches += 1

        return winner, looser

class Player(object):
    def __init__(self, name, score=100, wins=0, matches=0):
        self.name = name
        self.score = score
        self.wins = wins
        self.matches = matches
        self.ys = []
        self.index = []
        self.mu = 0
        self.sigma = 0
        self.ranki = Rating()
        self.mus = []
        self.sigmas = []
        self.pwin=[]
        self.plose=[]
        self.pdraw=[]

def assemble_dataframes(dir_name='./elo_temp/'):
    print('[RUN]: assembling dataframes')
    files = os.listdir(dir_name)
    dfs = []
    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])
    for dataframe in bar(files):
        df = pd.read_csv(dir_name+dataframe, index_col='Unnamed: 0')
        df.columns = [dataframe.strip('[].csv')]
        dfs.append(df)
    final = pd.concat(dfs, axis=0, ignore_index=False)

    return final

def get_unique_teams(df):
    ht = df.home_team.values
    at = df.away_team.values
    # create an array with both home and arway teams regardles of duplicates
    all_teams = np.append(ht, at)
    # drop all the duplicates
    all_teams = np.unique(all_teams)
    return all_teams

def compute_elo_by_goals2(scorer_func, data_df, players, all_teams):
    """
    This function is used to compute the elo ratings of teams based on simply wins and losses.
    :param scorer_func: func
    :param data_df: DataFrame
    :param players: list of classes
    :param all_teams: numpy array
    :return: list of classes
    """
    print('[FUNCTION]: computing elo by goals.')

    n_games = data_df.shape[0]

    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])

    ix = data_df.index
    for i in bar(range(n_games)[:]):
        match = data_df.iloc[i]
        player_home = match.home_team
        player_away = match.away_team
        hid = np.where(all_teams == player_home)[0][0]
        aid = np.where(all_teams == player_away)[0][0]
        pair = [players[hid], players[aid]]
        res = match.result_final
        home_goals = match.home_goals
        away_goals = match.away_goals

        if res == 0:
            for goal_difference in range(int(abs(home_goals-away_goals))):
                a, b = scorer_func(pair[0], pair[1])

            a.ys.append(a.score)
            b.ys.append(b.score)
            a.index.append(ix[i])
            b.index.append(ix[i])

        elif res == 2:
            for goal_difference in range(int(abs(home_goals-away_goals))):
                a, b = scorer_func(pair[1], pair[0])

            a.ys.append(a.score)
            b.ys.append(b.score)
            a.index.append(ix[i])
            b.index.append(ix[i])
        else:
            pair[0].ys.append(pair[0].score)
            pair[1].ys.append(pair[1].score)
            pair[0].index.append(ix[i])
            pair[1].index.append(ix[i])

    return players


def compute_elo_by_game_hdf5_v2(scorer_func, data_df, players, all_teams):
    print('[FUNCTION]: compute_elo_by_game_hdf5_v2')
    """
    This function is used to compute the elo ratings of teams based on simply wins and losses.
    :param scorer_func: func
    :param data_df: DataFrame
    :param players: list of classes
    :param all_teams: numpy array
    :return: list of classes
    """
    n_games = data_df.shape[0]
    elo_table = np.zeros((n_games, len(players)))
    # set initial score for all teams
    elo_table[0, :] = 100
    ix = data_df.index
    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])

    for i in bar(range(n_games)):
        # print('game: ', i)
        match = data_df.iloc[i]
        player_home = match.home_team
        player_away = match.away_team
        hid = np.where(all_teams == player_home)[0][0]
        aid = np.where(all_teams == player_away)[0][0]
        pair = [players[hid], players[aid]]
        res = match.result_final
        if res == 0:
            a, b = scorer_func(pair[0], pair[1])
            a.ys.append(a.score)
            b.ys.append(b.score)
            a.index.append(ix[i])
            b.index.append(ix[i])
        elif res == 2:
            a, b = scorer_func(pair[1], pair[0])

            a.ys.append(a.score)
            b.ys.append(b.score)
            a.index.append(ix[i])
            b.index.append(ix[i])
        else:
            pair[0].ys.append(pair[0].score)
            pair[1].ys.append(pair[1].score)
            pair[0].index.append(ix[i])
            pair[1].index.append(ix[i])

    return players

def save_players(players, pair_num, loc='./elo_temp/'):
    print('[SAVING]')
    h5f = h5py.File(loc + 'elo_pairs_'+str(pair_num)+'.h5', 'w')
    pair = 'pair_'+str(pair_num)
    h5f.create_group(pair)

    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])

    for pl in bar(players):
        name = pl.name
        index = pl.index
        ys = pl.ys
        h5f.create_dataset(pair + '/' + name, data=np.array([index, ys]))
    h5f.close()
    return None
