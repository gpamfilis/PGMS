
# http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017

from hmmlearn.hmm import MultinomialHMM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Here n_components correspond to number of states in the hidden
# variables and n_symbols correspond to number of states in the
# obversed variables
model_multinomial = MultinomialHMM(n_components=4)
# Transition probability as specified above
transition_matrix = np.array([[0.2, 0.6, 0.15, 0.05],
                                [0.2, 0.3, 0.3, 0.2],
                                [0.05, 0.05, 0.7, 0.2],
                                [0.005, 0.045, 0.15, 0.8]])

# Setting the transition probability
model_multinomial.transmat_ = transition_matrix
# Initial state probability
initial_state_prob = np.array([0.1, 0.4, 0.4, 0.1])
# Setting initial state probability
model_multinomial.startprob_ = initial_state_prob
# Here the emission prob is required to be in the shape of
# (n_components, n_symbols). So instead of directly feeding the
# CPD we would using the transpose of it.
emission_prob = np.array([[0.045, 0.15, 0.2, 0.6, 0.005],
                            [0.2, 0.2, 0.2, 0.3, 0.1],
                            [0.3, 0.1, 0.1, 0.05, 0.45],
                            [0.1, 0.1, 0.2, 0.05, 0.55]])
# Setting the emission probability

model_multinomial.emissionprob_ = emission_prob
# model.sample returns both observations as well as hidden states
# the first return argument being the observation and the second
# being the hidden states
# Z, X = model_multinomial.sample(5)
#
# print('X',X)
#
# print('Z',Z)

states = [0]#,1,2,3]
emisions = []

for state in states:
    em = []
    for i in range(100):
        em.append(model_multinomial._generate_sample_from_state(state)[0])
    emisions.append(em)    # print(state)
df = pd.DataFrame(emisions).T

df.plot(kind='hist');
plt.xticks([0,1,2,3,4]);
plt.show()
