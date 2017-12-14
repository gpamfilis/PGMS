# coding: utf-8

import pprint
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import Elo,Player, transition_matrix_pxx, emission_probabilities_pyx
from algorithms import  ForwardBackwardAlgorithm
from sklearn import preprocessing
from itertools import permutations

print('Reading Champs...')
champs = pd.read_csv('./champs.csv')
print('Reading Data...')
data = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')

def compute_elo_by_game():
    pass

def compute_elo_by_goals():
    pass

chs = range(51,100)
# chs = [20]
for ch in chs:
    eng1 = data[data.championship == champs.name[ch]]
    print('champ', eng1.shape)
    ht = eng1.home_team.values
    at = eng1.away_team.values
    all_teams = np.append(ht, at)
    all_teams = np.unique(all_teams)
    print(all_teams.shape)

    players = [Player(name=p) for p in all_teams]

    ngames = eng1.shape[0]
    elo = Elo()
    ss = np.zeros((ngames,len(players)))
    ss[0, :] = 100

    for i in range(ngames):
        print(i)
        match = eng1.iloc[i]
        p1 = match.home_team
        p2 = match.away_team
        hid = np.where(all_teams==p1)[0][0]
        aid = np.where(all_teams==p2)[0][0]
        pair = [players[hid], players[aid]]
        res = match.result_final
        if res==0:
            a, b = elo.match_algo_strict(pair[0], pair[1])
            ss[i, hid] = a.score
            ss[i, aid] = b.score
        elif res==2:
            a,b = elo.match(pair[1], pair[0])
            ss[i, aid] = a.score
            ss[i, hid] = b.score
        else:
            pass

    df = pd.DataFrame(ss,columns=all_teams)



    t = [(p.name, p.score) for p in players]
    nummatches= pd.DataFrame(t)
    cs = nummatches.sort_values(by=[1], ascending=False).iloc[:][0].values


    # for c in range(10):
    #     df[cs[c]] = df[cs[c]].replace(to_replace=0, method='ffill')
    #     df[cs[c]].dropna().plot()
    # plt.show()


    output = []






    for c, name in enumerate(cs[:]):

        px = eng1[eng1.home_team == cs[c]].result_final.value_counts(normalize=True).sort_index().values
        # pprint.pprint(px)

        pxx = transition_matrix_pxx(data=eng1[eng1.home_team == cs[c]], result='result_final', n_states=3)
        # pprint.pprint(pxx)

        # ### Emision Probs

        eng1.index = range(eng1.shape[0])

        ys = df[cs[c]].diff().replace(np.nan, 0).astype(int)[eng1[eng1.home_team == cs[c]].index]

        un = np.unique(ys.values)

        # try:

        pyx = emission_probabilities_pyx(ys=ys, x_df=eng1, x_label='result_final', h_states=3)
        # except Exception as e:
        #     print(e)
        #     continue

        le = preprocessing.LabelEncoder()

        le.fit(un)

        test = le.transform(ys)

        try:
            fba = ForwardBackwardAlgorithm(px, pxx, pyx, test)
            forward = fba.forward()
            backward = fba.backward()
            gammas = fba.gammas()
        except Exception as e:
            print(e)
            continue
        # pprint.pprint(np.argmax(gammas,axis=1))
        print(c, name)
        # ll = metrics.log_loss(eng1[eng1.home_team==cs[c]].result_final.values,gammas[:,np.argmax(gammas,axis=1)])
        # print('log loss: ', ll)
        try:
            ac = metrics.accuracy_score(eng1[eng1.home_team == cs[c]].result_final.values, np.argmax(gammas, axis=1))
            output.append([name, ac, eng1[eng1.home_team == cs[c]].result_final.values.shape[0]])
            print('accuracy: ', ac)
            # pprint.pprint(ac)
        except Exception as e:
            print(e)
            pass

    print('saving')
    df2 = pd.DataFrame(output)
    df2.to_csv('./data/yo_'+str(ch)+'.csv')

# df[cs[0]].diff().replace(to_replace=0, method='ffill')

# from __future__ import division
# import os
# from random import randint, choice
#
# MAX_INCREASE = 32
# INITIAL_SCORE = 1500
#
# class Player:
#
#     def __init__(self, id, score, skill, variation):
#         self.id = id
#         self.score = score
#         self.skill = skill
#         self.variation = variation
#         self.wins = 0
#         self.matches = 0
#
#     def get_score(self):
#         return self.skill + randint(-self.variation, self.variation)
#
#     def __str__(self):
#
#         if self.matches:
#             win_perc = self.wins / self.matches * 100
#         else:
#             win_perc = 0
#         return 'Player %3i %5i skill: %3i/%3i W:%3i %3.1f %%' % (self.id, self.score,
#                 self.skill, self.variation, self.wins, win_perc)
#
#     def __eq__(self, other):
#         return self.id == other.id
#
# class Elo:
#
#     def __init__(self, players):
#         self.players = [Player(a, INITIAL_SCORE, randint(0, 99), randint(0, 99)) for a in xrange(players)]
# #        self.players.append(Player(-1, INITIAL_SCORE, -500, 0))
#         self.players.append(Player(-2, INITIAL_SCORE, -50, 0))
#         self.players.append(Player(-3, INITIAL_SCORE, 0, 0))
# #        self.players.append(Player(-10, INITIAL_SCORE, 1500, 0))
#         self.players.append(Player(-11, INITIAL_SCORE, 500, 0))
#         self.players.append(Player(-12, INITIAL_SCORE, 100, 0))
#         self.output_match = True
#
#     def random_match(self):
#         p1 = choice(self.players)
#         p2 = p1
#         while p1 == p2:
#             p2 = choice(self.players)
#         self.match(p1, p2)
#
#     def match(self, p1, p2):
#         if self.output_match: print
#         e1 = MAX_INCREASE * 1 / (1 + 10 ** ((p2.score - p1.score) / 400))
#         e2 = MAX_INCREASE * 1 / (1 + 10 ** ((p1.score - p2.score) / 400))
#         s1 = p1.get_score()
#         s2 = p2.get_score()
#         if self.output_match: print p1, e1
#         if self.output_match: print ' vs'
#         if self.output_match: print p2, e2
#         p1.matches += 1
#         p2.matches += 1
#         if s1 > s2:
#             p1.wins += 1
#             p1.score += e2
#             p2.score -= e2
#             if self.output_match: print p1, 'won. Gained', e2
#             if self.output_match: print p2, 'loss. Lost ', e2
#         else:
#             p2.wins += 1
#             p1.score -= e1
#             p2.score += e1
#             if self.output_match: print p2, 'won. Gained', e1
#             if self.output_match: print p1, 'loss. Lost ', e1
#
#
#     def print_ranks(self):
#         print
#         s = sorted(self.players, key=lambda a: a.score)
#         last = 0
#         for player in s:
#             print player, player.id, int(player.score - last)
#             last = player.score
#
#
# def main():
#     e = Elo(5)
#     fast = True
#     e.output_match = not fast
#     matches = 0
#     while 1:
#         matches += 1
#         if matches % 100000 == 0:
#             e.players.append(Player(99, INITIAL_SCORE, randint(0, 99),
#                         randint(0, 99)))
#         e.random_match()
#         if fast and (matches % 100000):
#             continue
#         if fast:
#             os.system('clear')
#         e.print_ranks()
#         if not fast:
#             raw_input()
#
# if __name__ == '__main__':
#     main()
