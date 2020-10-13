"""
implementation of the classical and temporal gillespie algorithms
"""

import numpy as np
from scipy.optimize import fsolve


def gillespie(Q, x0, n_iter):
    x = x0
    xs = [x]
    ts = [0]
    for i in range(n_iter):
        rate = np.sum(Q[x,:]) - Q[x,x]
        tau = np.random.exponential(1/rate)
        q = Q[x,:] / rate
        q[x] = 0
        x = np.random.choice(range(len(q)), p=q)

        ts.append(ts[-1]+tau)
        xs.append(x)

    return np.array(xs), np.array(ts)


def temporal_gillespie(q_fun, int_fun, x0, t0, n_iter):
    """ gillespie algorithm for nonaut. processes """
    x = x0
    t = t0
    xs = [x]
    ts = [t]
    s = 1

    while len(xs) < n_iter:
        tau = np.random.exponential()
        t = fsolve(lambda tt: int_fun(t, tt, x) - tau, tau/s)[0]
        q = q_fun(t)[x, :]
        q[x] = 0
        s = np.sum(q)
        q = q / s
        x = np.random.choice(range(len(q)), p=q)
        xs.append(x)
        ts.append(t)

    return xs, ts


def temporal_gillespie_constant(qs, dts, x0, t0, n_iter):
    # constant generator on the timeinterval (0, dts]
    ts = np.cumsum(dts)

    def get_index(t):
        return next(i for i in range(len(ts)) if ts[i]>t)

    def q_fun(t):
        return qs[get_index(t)].toarray()

    def int_fun(s, t, x):
        if t<s:
            return 0
        if t>ts[-1]:
            return np.inf

        i_s = get_index(s)
        i_t = get_index(t)

        rates = 0

        for j in np.arange(i_s+1, i_t):
            rates += dts[j] * qs[j][x,x]

        rates += qs[i_s][x,x] * (ts[i_s] - s)
        rates += qs[i_t][x,x] * (t - ts[i_t-1])

        return np.exp(-rates)

    return temporal_gillespie(q_fun, int_fun, x0, t0, n_iter)
