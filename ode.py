""" 
ode finite time hitting probability computation 

we have

dq/dt (t,x) = - sum_y Q_xy(t) q(t,y) # in A'
dq/dt (t,x) = 0 # in A 
q(T) = 1_A # terminal condition

q(0,x) is the probability to hit set A in the finite timespan [0,T]

"""

from scipy.integrate import odeint
import numpy as np
from scipy.sparse.construct import spdiags

def finite_time_hitting_prob(Qs, dts, a, tol = 1e-8):
    """ fhtp for state a """

    nx = Qs[0].shape[0]
    x0 = np.zeros(nx)
    x0[a] = 1

    ts = np.cumsum(dts)

    def timeindex(t):
            return next((i for i in range(len(ts)) if ts[i]>t), len(ts)-1)

    def dq(x, t):
        ti = timeindex(ts[-1] - t)
        d = Qs[ti].dot(x)
        d[a] = 0
        d = np.nan_to_num(d)  # if the generator contains infinite rates, the matrix product results in inf-inf = nan
        return d

    return odeint(dq, x0, np.append([0], ts), rtol=tol, atol=tol)

def finite_time_hitting_probs(Qs, dts) -> np.ndarray:
    """ returns H[i,j] which is the fin. time hitting prob. to hit state i starting from j"""
    nx = Qs[0].shape[0]
    hps = np.array([finite_time_hitting_prob(Qs, dts, i)[-1,:] for i in range(nx)])
    return hps.T

class finite_time_hitting_prob_adjoint:
    def __init__(self, Qs, dts, us, nquad=100):
        self.Qs = Qs
        self.dts = dts
        self.cts = np.cumsum(dts)
        self.us = us

        self.nx = Qs[0].shape[0]
        self.nt = len(dts)

        self.nquad = nquad # integration stepsize
        self.its = np.linspace(0, self.cts[-1], nquad)

        self.dQs = [dqmultiplier(Qs[i], us[i].flatten()) for i in range(self.nt)]

        self.activei = None
        self.activej = None
        self.activey = None

    def timeindex(self, t):
        return next((i for i in range(self.nt) if self.cts[i]>t), len(self.cts)-1)

    def dfdy(self, t):
        # linear part of ode rhs: y' = f(y) = dfdy y
        return -self.Qs[self.timeindex(t)]

    def ode_y(self, j):
        # solve the finite time hitting ode for state j
        def dydt(y,t):
            d = self.dfdy(t).dot(y)
            d[j] = 0
            d = np.nan_to_num(d)
            return d
        
        yT = np.zeros(self.nx)
        yT[j] = 1

        y = odeint(dydt, yT, np.flip(self.its))
        y = np.flip(y, axis=0)
        return y
    
    def finite_time_hitting_probs(self):
        # compute all hitting probs and save the active minimium
        # hps[i,j] is the prob. to hit j when starting in i
        self.hps = np.zeros((self.nx, self.nx))
        self.hpmin = np.inf

        for j in range(self.nx):
            y = self.ode_y(j)
            y0 = y[0,:]
            cmin = np.min(y0)
            if cmin < self.hpmin:
                self.hpmin = cmin
                self.activej = j
                self.activei = np.argmin(y0)
                self.activey = y

            self.hps[:,j] = y0
        
        return self.hps

    def ode_mu(self):
        # adjoint ode for the derivative of the active hp

        def dmu(mu, t):
            mu[self.activej] = 0
            d = -self.dfdy(t).T.dot(mu)
            d = np.nan_to_num(d)
            return d

        mu0 = np.zeros(self.nx)
        mu0[self.activei] = 1

        self.mu = odeint(dmu, mu0, self.its)

    def dfdp(self, t, y):
        # df/dp = -dQ/dp * y
        return self.dQs[self.timeindex(t)](y)

    def adjointintegrate(self):
        # int_0^T mu^* f_p 

        its = self.its
        y = self.activey

        dg = np.zeros(self.nx)
        for i in range(len(its)-1):
            dt = its[i+1] - its[i]
            mu = self.mu[i]
            dqy = self.dfdp(its[i], y[i])
            #dg += -dt * np.dot(mu, dqy)
            dg += -dt * dqy.T.dot(mu)

        self.dg = dg

    def min_and_derivative(self):
        self.finite_time_hitting_probs()
        self.ode_mu()
        self.adjointintegrate()
        return self.hpmin, self.dg

import scipy.sparse as sp

def dqmultiplier(Q, u, beta=1):
    # return a function 'multiplier'
    # which in turn gives the derivative of the sqra Q=sqra(u) by u applied to y
    # i.e. multiplier = dqmultiplier(Q, u)
    # multiplier(y)[i,j] = (dQ/du_j * y)_i
 

    nx = Q.shape[0]

    rows =  beta/2 * Q.T   # rows[:,k] is the k-th row of the k-th derivative
    cols = -beta/2 * Q     # cols[:,k] is the k-th col of the k-th derivative
    diaginds = np.diag_indices_from(rows)
    rows[diaginds] = 0
    cols[diaginds] = 0
    diags = - cols
    diags[diaginds] += -np.sum(rows, axis=0)
    #diags = -np.sum(rows, axis=0)  # diags[:,k] is the diagonal of the k-th derivative

    def multiplier(y):
        d  = cols.dot(sp.spdiags(y, 0, nx, nx))
        #d += np.diag(y).dot(rows.T)
        d += sp.spdiags(y, 0, nx, nx).dot(rows.T)
        d += sp.spdiags(diags.dot(y), 0, nx, nx)
        return d
    
    return multiplier


class Rprop:
    def __init__(self, df, x0, g0=None, inc=1.2, dec=.5, max=50, min=10**(-6)):
        self.df = df
        self.x = x0
        self.lastdf = df(x0)
        if g0 is None:
            self.g = np.random.rand(len(x0))
        else:
            self.g = g0

        self.inc = inc
        self.dec = dec
        self.max = max
        self.min = min
        self.hist_x = [x0]

    def iterate(self):
        df = self.df(self.x)
        g = self.g
        for i in range(len(g)):
            if df[i] * self.lastdf[i] > 0:
                g[i] = np.minimum(g[i] * self.inc, self.max)
            else:
                g[i] = np.maximum(g[i] * self.dec, self.min)

        dx = - g * np.sign(df)
        self.lastdf = df
        self.x += dx
        self.hist_x.append(self.x.copy())

    def run(self, iter):
        for i in range(iter):
            self.iterate()

class Problem:
    def __init__(self, sqra, dts):
        self.sqra = sqra
        self.dts = dts

    def perturb(self, x):
        self.s = self.sqra.perturbed_copy(x)
        self.Qs = [self.s.Q] * len(self.dts)
        self.us = [self.s.u] * len(self.dts)
        self.hp = finite_time_hitting_prob_adjoint(self.Qs, self.dts, self.us)
    
    def obj(self, x):
        self.perturb(x)
        m, d = self.hp.min_and_derivative()
        print(-m)
        return m, d

    def objwithpenalty(self, x, p):
        m, d = self.obj(x)
        return m + p*np.sum(np.abs(x)), d + np.sign(x) * p
