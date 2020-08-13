import numpy as np
import pdb
import pylab as pl
from scipy.stats import poisson

# Params for the sigmoid activation
C = 1.0
L = 6.0
K = -0.5
t_0 = 15


# param for death reaction
delta = 0.1


def Birth_Reaction(x, t):
    return C + np.divide(L, 1+np.exp(-K*(t-t_0)))


def Death_Reaction(x, t):
    return delta*x


def Delta(t, s):  # need to test this
    return (L/K)*(np.log(1 + np.exp(K*(t-t_0))) - np.log(1+np.exp(K*(s-t_0)))) + C*(t-s)


def generate_Q_matrix(X, T):
    # formula (14,16) Q[i,j,t] = q~_ij(t) * q_i(t)
    Q = np.zeros((len(X), len(X), len(T)))
    # do it manuelly
    for t in range(len(T)):
        for x in range(len(X)):
            for y in range(len(X)):
                if y == x-1:  # from front
                    Q[x, y, t] += Death_Reaction(X[x], T[t])
                elif y == x+1:  # from behind
                    Q[x, y, t] += Birth_Reaction(X[x], T[t])
                elif x == y:
                    Q[x, y, t] = 0
                    '''
                    if x == 0:
                    Q[x,y,t] -=  Birth_Reaction(X[x],T[t])
                    else:
                    Q[x,y,t] -=  Birth_Reaction(X[x],T[t]) + Death_Reaction(X[x],T[t])
                    '''

        # Q[-1, :, t] = 0  # no outflow.

    # Q[-1, -1, :] = 1  # boundary condition
    return Q


# new formulation
# using veriable names as in https://arxiv.org/abs/2008.04624

def generate_S(X, T):
    """ S[i,s,t] = exp(- int_s^t q_i)  [eq. 24] """
    S = np.zeros((len(X), len(T), len(T)))

    for t in range(len(T)):
        for s in range(t+1):
            int_birth = Delta(T[t], T[s])
            for i in range(len(X)):
                int_death = delta*X[i]*(T[t]-T[s])
                S[i, s, t] = np.exp(-(int_birth + int_death))

    return S


def jump(Q, S, p):
    # based on formula (16)
    # but using the trick that qt*qi is almost Q (except for diagonal and 0-rates)
    return np.einsum('ijt, ist, is -> jt', Q, S, p)


def test_doessame():
    delta_t = 0.1

    X = np.arange(0, 20, 1)
    T = np.arange(0, 20, delta_t)

    Q = generate_Q_matrix(X, T)
    I, Tau = generate_I_and_Tau(X, T, delta_t)

    S = generate_S(X, T)

    p0 = np.zeros((len(X), len(T)))
    p0[:, 0] = poisson.pmf(X, mu=10)

    pp1 = propogate(Q, I, Tau, p0)
    pp2 = jump(Q, S, p0)
    pp3 = jump(Q, S*delta_t, p0)

    assert np.isclose(pp1, pp3).all()


def generate_I_and_Tau(X, T, delta_T):
    I = np.zeros((len(X), len(T), len(T)))
    Tau = np.zeros((len(T), len(T)))

    for x in range(len(X)):
        for t in range(len(T)):
            for s in range(t+1):  # s <= t
                # why do we index t,s and not s,t?
                I[x, t, s] = np.exp(-delta*X[x]*(T[t]-T[s]))
                if x == 0:
                    Tau[t, s] = np.exp(-Delta(T[t], T[s]))

    return I*delta_T, Tau  # alex: why *delta_T?


def propogate(Q, I, Tau, p_0):

    F = np.einsum('xts, ts, xs -> xt', I, Tau, p_0)

    p_1 = np.einsum('xyt, xt -> yt', Q, F)

    return p_1



def test_run(nx=50, nt=40, delta_t=0.1):

    X = np.arange(0, nx, 1)
    T = np.arange(0, nt, delta_t)

    Q = generate_Q_matrix(X, T)
    I, Tau = generate_I_and_Tau(X, T, delta_t)

    S = generate_S(X, T, rownormalize=True)

    p_0 = np.zeros((len(X), len(T)))
    p_0[:, 0] = poisson.pmf(X, mu=10)

    p_X_T = p_0.copy()

    '''
    p_1 = propogate(Q,I,Tau,p_0)
    pl.subplot(1,2,1)
    pl.imshow(p_0, origin = 'low', aspect = 'auto')
    pl.colorbar()
    pl.subplot(1,2,2)
    pl.imshow(p_1, origin = 'low', aspect = 'auto')
    pl.colorbar()
    pl.show()
    '''

    pl.ion()

    pp = p_0
    for i in range(200):

        pp = jump(Q, S, pp)

        pl.clf()
        pl.title('step:' + str(i))
        p_X_T += pp

        print(np.sum(pp))

        pl.imshow(pp.T, origin='lower', aspect='auto')
        #pl.imshow(p_X_T.T, origin = 'low', aspect = 'auto')

        pl.colorbar()
        inds = range(len(T))[::40]

        pl.yticks(inds, T[inds])
        pl.xlabel('X')
        pl.ylabel('Time')

        pl.draw()
        pl.pause(0.1)

    pdb.set_trace()


if __name__ == '__main__':
    test_run()
