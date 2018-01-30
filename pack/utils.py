from itertools import permutations
import numpy as np
import pandas as pd
import progressbar

K = 10

def test_import():
    print('test')


def string_list_to_numpy(x):
    x = x.strip('[] ').split()
    ar = np.asarray(x).astype(float)
    return ar


def emission_probabilities_pyx(y_df, x_df, x_label='over_under_2.5', hidden_states=2):
    un = np.unique(y_df.values)
    pyx = np.zeros((hidden_states, np.unique(y_df.values).shape[0]))
    # print(un,pyx)
    for i, ix in enumerate(y_df.index):
        try:
            yx = np.where(y_df.loc[ix, x_label] == un)[0][0]
            pyx[int(x_df.loc[ix, x_label]), int(yx)] += 1
        except Exception as e:
            # print('exception',e)
            continue
    for i, s in enumerate(pyx.sum(axis=1)):
        pyx[i, :] = pyx[i, :] / s
    return pyx






def string_list_to_numpy(x):
    x = x.strip('[] ').split()
    ar = np.asarray(x).astype(float)
    return ar




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
    print(all_teams.shape)
    # create a a Player class for each team. this class stores the elo rating for each player
    players = [Player(name=p) for p in all_teams]
    elo = Elo()

    # elo_table_array = compute_elo_by_game(matches_champ, players, all_teams, elo)
    elo_table_array = compute_elo_by_goals(matches_champ, players, all_teams, elo)

    team_elo_timeline_df = pd.DataFrame(elo_table_array, columns=all_teams)
    return team_elo_timeline_df

class Elo(object):

    def match(self, p1, p2):
        return self.match_algo_strict(p1, p2)

    @staticmethod
    def match_algo_strict(winner, looser):
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



def transition_matrix_pxx(data, result, n_states=None):
    """
    this method computes the transition matrix for single states NOT paired states.
    """
    if n_states is None:
        states = data[result].dropna().drop_duplicates().sort_values().astype(int).values
    else:
        states = range(n_states)
    states_same_index = []
    for state in states:
        states_same_index.append((state, state))
    data_df = data[result].dropna()
    # stupid but works
    transitions = [(int(i), int(j)) for i, j in list(permutations(states, 2)) + states_same_index]

    transition_matrix = np.zeros((len(states), len(states)))
    for index, transition in enumerate(transitions):
        for i in range(data_df.shape[0]-1):
            if (data_df.iloc[i], data_df.iloc[i + 1]) == transition:  # automatically create a tuple for multiple states
                transition_matrix[transition] += 1
    for state in states:
        transition_matrix[state, :] = transition_matrix[state, :]/transition_matrix[state, :].sum()
    return transition_matrix


# if __name__ == '__main__':
#     elo = Elo()
#     p1 = Player(name=0)
#     p2 = Player(name=1)
#     elo.match(p1, p2)


def compute_elo_by_game(data_df, players, all_teams, elo):
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
            a, b = elo.match_algo_strict(pair[0], pair[1])
            elo_table[i, hid] = a.score
            elo_table[i, aid] = b.score
        elif res == 2:
            a, b = elo.match(pair[1], pair[0])
            elo_table[i, aid] = a.score
            elo_table[i, hid] = b.score
        else:
            pass

    return elo_table


def compute_elo_by_goals(data_df, players, all_teams, elo, initial_score=100):
    """
    This function is used to compute the elo ratings of teams based on goals scored depending on wins and losses.

    :param data_df:
    :param players:
    :param elo:
    :return:
    """
    n_games = data_df.shape[0]
    elo_table = np.zeros((n_games, len(players)))
    # set initial score for all teams
    elo_table[0, :] = initial_score
    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])
    for i in bar(range(n_games)):

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
                a, b = elo.match_algo_strict(pair[0], pair[1])
            elo_table[i, hid] = a.score
            elo_table[i, aid] = b.score
        elif res == 2:
            for goal_difference in range(int(abs(home_goals-away_goals))):
                a, b = elo.match(pair[1], pair[0])
            elo_table[i, aid] = a.score
            elo_table[i, hid] = b.score
        else:
            pass

    # make continuous
    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])
    print('[C] Making continuous')
    for p in bar(range(len(players))):

        for g in range(n_games):
            if elo_table[g, p] == 0:
                elo_table[g, p] = elo_table[g-1, p]
            else:
                pass
    return elo_table
