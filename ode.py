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

def finite_time_hitting_prob(Qs, dts, a):
    """ fhtp for state a """

    nx = Qs[0].shape[0]
    nt = len(Qs)
    x0 = np.zeros(nx)
    x0[a] = 1

    ts = np.cumsum(dts)

    def timeindex(t):
            return next((i for i in range(len(ts)) if ts[i]>t), len(ts)-1)

    def dq(x, t):
        ti = timeindex(ts[-1] - t)
        d = Qs[ti].dot(x)
        d[a] = 0
        return d

    return odeint(dq, x0, np.append([0], ts))

def finite_time_hitting_probs(Qs, dts):
    """ returns H[i,j] which is the fin. time hitting prob. to hit state i starting from j"""
    nx = Qs[0].shape[0]
    hps = np.array([finite_time_hitting_prob(Qs, dts, i)[-1,:] for i in range(nx)])
    return hps.T
