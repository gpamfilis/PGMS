import numpy as np
from hmmlearn import hmm

states = ["Rainy", "Sunny"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

model = hmm.MultinomialHMM(n_components=n_states, init_params="")
model.startprob_ = np.array([0.6, 0.4])
model.transprob_ = np.array([[0.7, 0.3],[0.4, 0.6]])
model.emissionprob_ = np.array([[0.1, 0.4, 0.5],[0.6, 0.3, 0.1]])
# predict a sequence of hidden states based on visible states
bob_says = np.array([[0, 2, 1, 1, 2, 0]]).T

model = model.fit(bob_says)
logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
print("Bob says:", ", ".join(map(lambda x: observations[int(x)], bob_says)))
print("Alice hears:", ", ".join(map(lambda x: states[x], alice_hears)))
