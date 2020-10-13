"""
joint work with vikram

a simple birth-death model
approximated with the augmented jump chain (with the collocation discretization)
and compared to the classical evolution
"""


import numpy as np
import pdb
import pylab as pl
import matplotlib.pyplot as plt
from scipy.stats import poisson
import copy

import ajc
import gillespie


class BirthDeath:

    def __init__(self, C=1.0, L=1.0, K=-.5, t_0=15, delta=.1, n_x=40, n_t=100, t_max=40, auto_gen=True):
        self.C = C
        self.L = L
        self.K = K
        self.t_0 = t_0
        self.delta = delta  # death rate
        self.delta_t = 0.1  # stepsize of time discretization
        self.X = range(n_x)
        self.T = np.linspace(0, t_max, n_t)

        self.p0 = np.zeros((n_x, n_t))
        self.p0[:, 0] = poisson.pmf(self.X, mu=10)

        if auto_gen:
            self.generate_Q()
            self.generate_S()
            self.generate_ajc()

    def Birth_Reaction(self, x, t):
        return self.C + np.divide(self.L, 1+np.exp(-self.K*(t-self.t_0)))

    def Death_Reaction(self, x, t):
        return self.delta*x

    def Delta(self, t, s):
        # integral of the logistic birth function from s to t
        return (self.L/self.K)*(np.log((1 + np.exp(self.K*(t-self.t_0))) / (1+np.exp(self.K*(s-self.t_0))))) + self.C*(t-s)

    def generate_Q(self):
        X, T = self.X, self.T
        Q = np.zeros((len(X), len(X), len(T)))
        nx = len(X)
        for t in range(len(T)):
            for x in range(len(X)):
                if x - 1 >= 0:
                    Q[x, x-1, t] += self.Death_Reaction(X[x], T[t])
                if x + 1 < nx:
                    Q[x, x+1, t] += self.Birth_Reaction(X[x], T[t])
        self.Q = Q

    def generate_S(self):
        """ S[i,s,t] = exp(- int_s^t q_i)  [eq. 24] """
        X, T = self.X, self.T

        S = np.zeros((len(X), len(T), len(T)))
        s, t = np.meshgrid(T, T)

        birth = self.Delta(s, t)
        S[None, :, :] = birth.T

        death = s - t
        death = self.delta * np.einsum('st, i -> ist', death, X)

        S = np.exp(-(birth + death))
        S = np.triu(S)

        self.S = S

    def generate_ajc(self):
        self.ajc = ajc.AJCCollocation(self.Q, self.S)

    def gillespie(self, x0, t0, n):
        bd = copy.copy(self)

        def q_fun(t):
            bd.T = [t]
            bd.generate_Q()
            return bd.Q[:, :, 0]

        def int_fun(s, t, i):
            return (bd.Delta(t, s) + bd.delta * (t-s) * bd.X[i])

        xs, ts = gillespie.temporal_gillespie(q_fun, int_fun, x0, t0, n)
        return xs, ts





def my_imshow(x, t_max=40, x_max=40):
    plt.imshow(x, origin='lower', aspect='auto', extent=[0, t_max, 0, x_max])
    plt.xlabel('T')
    plt.ylabel('X')
    #plt.colorbar()


def plot_activity(xs):
    xsum = np.sum(xs, axis=0)
    #print(np.argmax(np.sum(xsum, axis=1)))
    my_imshow(xsum)


def run(p=BirthDeath(), n_jumps=100, n_gillespie=100):
    """ the experiment """
    x = p.p0
    xs = [x]

    gxs = np.zeros((n_gillespie, n_jumps))
    gts = np.zeros((n_gillespie, n_jumps))

    for i in range(n_gillespie):
        x0 = np.random.choice(range(len(p.X)), p=p.p0[:, 0])
        gxs[i, :], gts[i, :] = p.gillespie(x0, 0, n_jumps)
        plt.plot(gts, gxs)

    pl.ion()

    for i in range(n_jumps):

        x = p.ajc.jump(x) ### JUMP CHAIN
        xs.append(x)

        pl.clf()
        pl.title('step:' + str(i))
        my_imshow(x)
        x0 = np.random.choice(range(len(p.X)), p=p.p0[:, 0])
        plt.scatter(gts[:, i], gxs[:, i])

        pl.draw()
        pl.pause(0.01)

    plt.figure()
    plot_activity(xs)
    plt.plot(gts.T, gxs.T, alpha=.1, color="white")
    plt.title("AJC vs Gillespie")

    return locals()


if __name__ == '__main__':
    r = run()
    pdb.set_trace()
