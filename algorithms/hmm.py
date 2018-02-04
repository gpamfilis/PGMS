import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

class SumProduct(object):
    pass

class MaxProduct(object):
    pass

class ViterbyAlgorithm(object):
    pass

class ForwardBackwardAlgorithm(object):
    def __init__(self, px, pxx, pyx, y):
        self.px=px
        self.pxx=pxx
        self.pyx=pyx
        self.y = y
        self.hidden_states=range(self.px.shape[0])

    def ai_line(self, ai_ar, t, xi, s=0):
        return ai_ar[t-1, s]* self.pxx[s, xi] * self.pyx[s, self.y[t]]

    def bi_line(self, ai_ar, t, xi, s=0):
        return bi_ar[t+1, s] * self.pxx[xi, s] * self.pyx[s, self.y[t+1]]

    def forward(self):
        """
        The forward part of the Algorithm.

        todo: insert math expression
        """
        ais = []

        for state in self.hidden_states:
            a = self.pyx[state, self.y[0]] * self.px[state]
            ais.append(a)

        ai_ar = np.zeros((len(self.y), len(self.hidden_states)))

        ai_ar[0, :] = ais

        for t in range(1, len(self.y)):
            for xi in self.hidden_states:
                # ai_ar[t, xi] = (ai_ar[t-1, 0]* self.pxx[0, xi] * self.pyx[0, self.y[t]]) + (ai_ar[t-1, 1] * self.pxx[1, xi] * self.pyx[1, self.y[t]])
                for s in self.hidden_states:
                        ai_ar[t, xi] += (ai_ar[t-1, s] * self.pxx[s, xi] * self.pyx[s, self.y[t]])
        return ai_ar

    def backward(self):
        bi = []
        for state in self.hidden_states:
            bi.append(1)
        bi_ar = np.zeros((len(self.y), len(self.hidden_states)))
        bi_ar[-1,:] = bi
        for t in np.arange(len(self.y)-2,-1,-1):
            for xi in self.hidden_states:
                # bi_ar[t,xi] = (bi_ar[t+1, 0] * self.pxx[xi, 0] * self.pyx[0, self.y[t+1]]) + (bi_ar[t+1, 1] * self.pxx[xi, 1] * self.pyx[1, self.y[t+1]])
                for s in self.hidden_states:
                    bi_ar[t,xi] += (bi_ar[t+1, s] * self.pxx[xi, s] * self.pyx[s, self.y[t+1]])

        return bi_ar

    def gammas(self):
        ai = self.forward()
        bi = self.backward()
        l = []
        for t in range(len(self.y)):
            ab = (ai[t] * bi[t])
            s = ab.sum()
            d = ab/s
            l.append(d)
        out = np.zeros((len(self.y), len(self.hidden_states)))
        for i, a in enumerate(l):
            out[i,:] = a
        return out

if __name__ == '__main__':
    mat = scipy.io.loadmat('./TΕL606_labnotes_3.mat')
    data = mat['price_move']
    l =  data.shape[0]
    price_move = np.zeros(l)
    for i, p in enumerate(data):
        price_move[i] = p[0]
    ys = price_move
    df = pd.DataFrame(price_move,columns=['price_move'])

    q = 0.9
                #  G    B
    px = np.array([0.2,0.8])
                #    G   B
    pxx = np.array([[.8,.2],
                    [.2,.8]])
                  #  -1   +1
    pyx = np.array([[1-q, q],
                    #G
                    #B
                    [q,1-q]])

    ytra = []
    for i in ys:
        if i==-1:
            ytra.append(0)
        else:
            ytra.append(1)

    hidden_states=[0,1]


    fba = ForwardBackwardAlgorithm(px,pxx,pyx,ytra)
    forward = fba.forward()
    backward = fba.backward()
    gammas = fba.gammas()
    plt.plot(gammas[:,0])
    plt.plot(gammas[:,1])

    plt.plot(ys)
    plt.show()
