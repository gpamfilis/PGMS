import numpy as np


class SumProduct(object):
    pass

class MaxProduct(object):
    pass

class ViterbyAlgorythm(object):
    pass


class ForwardBackwardAlgorythm(object):
    def __init__(self, px, pxx, pyx, y):
        self.px=px
        self.pxx=pxx
        self.pyx=pyx
        self.y = y
        self.hidden_states=range(self.px.shape[0])

    def forward(self):
        """
        The forward part of the algorythm.

        todo: insert math expression
        """
        a1g = self.pyx[self.y[0],0]*self.px[0]#(1-q) * 0.2
        a1b = self.pyx[self.y[0],1]*self.px[1]#q * .8
        ai_ar = np.zeros((len(self.y), 2))
        ai_ar[0, :] = [a1b, a1g]
        for t in range(1, len(self.y)):
            for xi in self.hidden_states:
                ai_ar[t, xi] = (ai_ar[t-1, 0]* self.pxx[0, xi] * self.pyx[0, self.y[t]]) +(ai_ar[t-1, 1] * self.pxx[1, xi] * self.pyx[1, self.y[t]])
        return ai_ar

    def backward(self):
        b1g = 1
        b1b = 1
        bi_ar = np.zeros((len(self.y), 2))
        bi_ar[-1,:] = [b1b, b1g]
        for t in np.arange(len(self.y)-2,-1,-1):
            for xi in self.hidden_states:
                bi_ar[t,xi] = (bi_ar[t+1, 0] * self.pxx[xi, 0] * self.pyx[0, self.y[t+1]]) + (bi_ar[t+1, 1] * self.pxx[xi, 1] * self.pyx[1, self.y[t+1]])
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
        out = np.zeros((len(self.y), 2))
        for i, a in enumerate(l):
            out[i,:] = a
        return out
