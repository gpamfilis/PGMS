# -*- coding: utf-8 -*-

import os
import pprint
from trueskill import Rating
from trueskill import rate_1vs1

import progressbar

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import metrics

import matplotlib.pyplot as plt

def get_metrics(clf,X,y):
    y_pred = clf.predict(X)
    accuracy = metrics.accuracy_score(y_pred, y)
    n_classes = np.unique(y_pred)
    n_classes.sort()
    pres = []
    for p in n_classes:
        pres.append(metrics.precision_score(y_pred, y, pos_label=p))
    reca = []
    for p in n_classes:
        reca.append(metrics.recall_score(y_pred, y, pos_label=p))
    classification_report = metrics.classification_report(y_pred, y)
    pprint.pprint(accuracy)
    pprint.pprint(metrics.classification_report(y_pred, y))
    return accuracy, pres

def filter_out_100(data):
    print('[Filter]')
    dataf = data[(data.EH!=100) & (data.EA!=100)].iloc[:]
    return dataf

def plot_data_labels(data, labels=['1','X','2'], npoints=1000):
    x = data.iloc[:npoints]['EH'].values
    y = data.iloc[:npoints]['EA'].values
    label = data.iloc[:npoints][result_key].values
    colors = ['green','blue','black']

    fig = plt.figure(figsize=(8,8))
    plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('EH')
    plt.ylabel('EA')
    cb = plt.colorbar()
    loc = np.arange(0,max(label),max(label)/float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)
    return data

def plot_data_labels_win_loose(data, labels=['1','X','2'], npoints=1000):
    x = data.iloc[:npoints]['WinProb'].values
    y = data.iloc[:npoints]['DrawProb'].values
    label = data.iloc[:npoints][result_key].values
    colors = ['green','blue','black']

    fig = plt.figure(figsize=(8,8))
    plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('WinProb')
    plt.ylabel('DrawProb')
    cb = plt.colorbar()
    loc = np.arange(0,max(label),max(label)/float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)
    return data

def run_pca(data, cols=[7,8]):
    print('[PCA]')

    pca = PCA(n_components=2)

    # print(pca.explained_variance_ratio_)  

    # print(pca.singular_values_)  

    pca.fit(data.values[:,cols])

    tx =  pca.transform(data.values[:,cols]) 

    data.loc[:,['EH','EA']] = tx
    return data

def classify(data):
    print(data.head())
    print('[Classifier]')
    X = data.iloc[:][['home_team','away_team','EH','EA']]#.values
    # X = final.iloc[:][['EH','EA']]#.values
    y = data.iloc[:][result_key]#.values

    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.75, test_size=0.25, random_state=42)

    # final.iloc[:50000].tail()

    clf = GaussianNB()
    clf = clf.fit(X_train, y_train)

    print('Accuracy: ',clf.score(X_test, y_test))
    # print(metrics.accuracy_score(clf.predict(X_test), y_test))
    # clf.export('tpot_mnist_pipeline.py')

    predictions = clf.predict(X_test)

    pprint.pprint(metrics.classification_report(predictions, y_test))

#     predicted = cross_val_predict(clf, X,y, cv=5)
    return data

def pipeline_func(data, fns):
    return reduce(lambda a,x: x(a), fns, data)

def get_team_data(p, pair):
    h5f = h5py.File('./elo_temp_goals/elo_pairs_'+str(p)+'.h5','r')
    champ = list(h5f.keys())[0]
    teams = list(h5f[champ].keys())
    teamh = df_teams[p,0]
    teama = df_teams[p,1]
    datah = h5f[champ][teamh][:]
    dataa = h5f[champ][teama][:]
    h5f.close()
    return champ, teams, teamh, teama, datah, dataa

def get_ratings(datah, dataa, teamh, teama):
    ht = pd.DataFrame(datah[1], columns=[teamh], index=datah[0].astype(int))
    at = pd.DataFrame(dataa[1], columns=[teama], index=dataa[0].astype(int))
    last_score_home, last_score_away = ht.iloc[-1].values[0], at.iloc[-1].values[0]
    ht.iloc[1:] = ht.values[0:-1]
    ht.iloc[0] = 100
    at.iloc[1:] = at.values[0:-1]
    at.iloc[0] = 100
    htd = ht.diff().replace(np.nan, 0).astype(int)
    atd = at.diff().replace(np.nan, 0).astype(int)
    return ht, at, htd, atd, last_score_home, last_score_away

def get_data_home(df, htd, result_key):
    x_df_home = df.loc[htd.index]
    x_df_home = x_df_home[x_df_home.home_team==htd.columns[0]]
    x_home = x_df_home[result_key].dropna().values
    ix_home = x_df_home.index.values
    y_home = htd.loc[ix_home].values.reshape(-1)
    data = np.array(list(zip(ix_home, x_home, y_home)))
    return data

def get_data_away(df,atd,result_key):
    x_df_away = df.loc[atd.index]
    x_df_away = x_df_away[x_df_away.away_team==atd.columns[0]]
    x_away = x_df_away[result_key].dropna().values
    ix_away = x_df_away.index.values
    y_away = atd.loc[ix_away].values.reshape(-1)
    data = np.array(list(zip(ix_away, x_away, y_away)))
    return data

def get_data_home_versus_all(df, htd, result_key):
    ix_Hall = htd.index.values
    x_Hall = df.loc[htd.index][result_key].dropna().values
    y_Hall = htd.values.reshape(-1)

    data = np.array(list(zip(ix_Hall, x_Hall, y_Hall)))
    return data

def get_data_away_versus_all(df, atd, result_key):
    ix_Aall = atd.index.values
    x_Aall = df.loc[atd.index][result_key].dropna().values
    y_Aall = atd.values.reshape(-1)
    data = np.array(list(zip(ix_Aall, x_Aall, y_Aall)))
    return data


def get_ids_and_mising(f_name='./slavesCombine/trueskill_temp/'):
    files = os.listdir(f_name)
    # https://stackoverflow.com/questions/20718315/how-to-find-a-missing-number-from-a-list
    a = [int(f.split('_')[-1].split('.')[0]) for f in files]
    return a, list(set(range(a[len(a)-1])[1:]) - set(a))

def rewrite_data(pair,data):
    """
    The data must be shifted
    """
    ht = data[pair[0]][:].T 
    at = data[pair[1]][:].T
    
    ht[1:, 1] = ht[0:-1, 1]
    ht[0, 1] = 25
    ht[0, 2] = 8.333
    
    at[1:, 1] = at[0:-1, 1]
    at[0, 1] = 25
    at[0, 2] = 8.333
    return ht, at

def get_trueskill_parameters_home_and_away(data_h5, row, ix):
    """
    This functions takes the data from the pair. iterates over the main data
    and finds the coresponding elo ratings based on the side the team is on.
    """
#     hdhome = data_h5[row.home_team][:].T
#     hdaway = data_h5[row.away_team][:].T
    pair = [row.home_team,row.away_team]
    hdhome,hdaway = rewrite_data(pair, data_h5)
    # home_ix = np.where(hdhome==ix)[1]
    # away_ix = np.where(hdaway==ix)[1]
    home_ix = np.where(hdhome[:,0]==ix)[0][0]
    away_ix = np.where(hdaway[:,0]==ix)[0][0]

    true_skill_home_mu = hdhome[home_ix,1]
    true_skill_home_sigma = hdhome[home_ix,2]

    true_skill_away_mu = hdaway[away_ix,1]
    true_skill_away_sigma = hdaway[away_ix,2]

    return true_skill_home_mu,true_skill_home_sigma, true_skill_away_mu,true_skill_away_sigma

def get_last_trueskill_data(pair,data_h5):
    home_team = pair[0]
    away_team = pair[1]
    home_data = data_h5[home_team][:]
    away_data = data_h5[away_team][:]
    hmu, hsigma, amu, asigma = home_data[-1, 1], home_data[-1, 2], away_data[-1, 1], home_data[-1, 2]
    return hmu,hsigma,amu,asigma

def get_match_indexes(data_h5, teams):
    """
    This function returns the unique index for the pairs.
    """
    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])
    arrays = []
    for team in bar(teams[:]):
        ix = data_h5[team][:].T[:,0]
        arrays.append(ix)
    uni = np.unique(np.concatenate(arrays)).astype(int)
    return uni

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

def compute_trueskill_by_game(data_df, players, all_teams):
    """
    This function is used to compute the elo ratings of teams based on simply wins and losses.
    :param data_df:
    :param players:
    :param elo:
    :return: array
    """
    n_games = data_df.shape[0]
    # elo_table = np.zeros((n_games, len(players)))
    # set initial score for all teams
    # elo_table[0, :] = 100
    ix = data_df.index

    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])

    for i in bar(range(n_games)[:]):
        match = data_df.iloc[i]
        player_home = match.home_team
        player_away = match.away_team
        hid = np.where(all_teams == player_home)[0][0]
        aid = np.where(all_teams == player_away)[0][0]
        pair = [players[hid], players[aid]]
        res = match.result_final
        if res == 0:
            a, b = rate_1vs1(pair[0].ranki, pair[1].ranki)

            pair[0].ranki = a
            pair[1].ranki = b

            pair[0].mus.append(a.mu)
            pair[0].sigmas.append(a.sigma)

            pair[1].mus.append(b.mu)
            pair[1].sigmas.append(b.sigma)

            pair[0].index.append(ix[i])
            pair[1].index.append(ix[i])
        elif res == 2:
            a, b = rate_1vs1(pair[1].ranki, pair[0].ranki)
            # print(a,b)
            pair[1].ranki = a
            pair[0].ranki = b

            pair[1].mus.append(a.mu)
            pair[1].sigmas.append(a.sigma)

            pair[0].mus.append(b.mu)
            pair[0].sigmas.append(b.sigma)

            pair[1].index.append(ix[i])
            pair[0].index.append(ix[i])
        else:
            a, b = rate_1vs1(pair[1].ranki, pair[0].ranki, drawn=True)

            pair[1].ranki = a
            pair[0].ranki = b

            pair[1].mus.append(a.mu)
            pair[1].sigmas.append(a.sigma)

            pair[0].mus.append(b.mu)
            pair[0].sigmas.append(b.sigma)

            pair[1].index.append(ix[i])
            pair[0].index.append(ix[i])

    return players

def save_players_trueskill(players, pair_num, loc='./elo_temp/'):

    print('[SAVING]')
    h5f = h5py.File(loc + 'trueskill_pairs_'+str(pair_num)+'.h5', 'w')
    pair = 'pair_'+str(pair_num)
    h5f.create_group(pair)

    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])

    for pl in bar(players[:]):
        name = pl.name
        index = pl.index
        mus = pl.mus
        sigmas =pl.sigmas
        h5f.create_dataset(pair + '/' + name, data=np.array([index, mus, sigmas]))
    h5f.close()
    return None

def get_match_indexes(data_h5, teams):
    """
    This function returns the unique index for the pairs.
    """
    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])

    arrays = []
    for team in bar(teams[:]):
        ix = data_h5[team][:].T[:,0]
        arrays.append(ix)
    uni = np.unique(np.concatenate(arrays)).astype(int)
    return uni
