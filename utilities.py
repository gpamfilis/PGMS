from itertools import permutations
import numpy as np

K = 10

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
    def __init__(self,name,score=100, wins=0, matches=0):
        self.name=name
        self.score=score
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


def emission_probabilities_pyx(ys, x_df, x_label='result_final', h_states=3):
    un = np.unique(ys.values)
    pyx = np.zeros((h_states, np.unique(ys.values).shape[0]))
    for i, ix in enumerate(ys.index):
        try:
            yx = np.where(ys[ix] == un)[0][0]
            pyx[int(x_df.loc[ix][x_label]), int(yx)] += 1
        except Exception as e:
            print(e)
            continue
    for i, s in enumerate(pyx.sum(axis=1)):
        pyx[i, :] = pyx[i, :] / s
    return pyx

if __name__ == '__main__':
    elo = Elo()
    p1 = Player(name=0)
    p2 =Player(name=1)
    elo.match(p1, p2)
