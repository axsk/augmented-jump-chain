import numpy as np
import pdb
import pylab as pl
import matplotlib.pyplot as plt
from scipy.stats import poisson


class BirthDeath:

    def __init__(self, C=1.0, L=6.0, K=-.5, t_0=15, delta=.1, n_x=40, n_t=40, t_max=40, auto_gen=True):
        self.C = C
        self.L = L
        self.K = K
        self.t_0 = t_0
        self.delta = delta  # death rate
        self.delta_t = 0.1  # stepsize of time discretization
        self.X = range(n_x)
        self.T = np.linspace(0, t_max, n_t)

        self.p_0 = np.zeros((n_x, n_t))
        self.p_0[:, 0] = poisson.pmf(self.X, mu=10)

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
        self.ajc = AugmentedJumpChain(self.Q, self.S)


class AugmentedJumpChain:
    def __init__(self, Q, S):
        self.Q = Q
        self.qt, self.qi = qtilde(Q)
        self.S = S
        self.k = jumpkernel_coll(self.qt, self.qi, self.S)
        nxt = Q.shape[0] * Q.shape[2]
        self.km = self.k.reshape(nxt, nxt).T

    def jump(self, p):
        return np.tensordot(self.k, p, ([0, 1], [0, 1]))  # sum over first two inds

    def jump2(self, p):
        nx, nt = p.shape
        return self.km.dot(p.reshape(nx*nt)).reshape(nx, nt)

    def jump3(self, p):
        return np.einsum('isjt, is -> jt', self.k, p)

    #def synchronize


def jumpkernel_coll(qt, qi, S):
    """
    compute the jumpkernel for given integrals S[i,s,t] = exp(-int_s^t q_i)
    this is the collocation approach, i.e. take the density of k(i,s,j,t) as a
    transtion probability (i,s) -> (j,t).
    """
    k = np.einsum('ijt, it, ist -> isjt', qt, qi, S)
    k = k / k.sum(axis=(2, 3))[:, :, None, None]  # normalize density to probability
    return k


def qtilde(Q):
    """ given a standard rate matrix returns:
    qt[i,j,t] = q^tilde_ij(t) : the row normalized jump matrix [eq. 14]
    qi[i,t] = q_i(t): the outflow rate [eq. 6]
    """

    qt = Q.copy()
    n = qt.shape[0]
    qt[range(n), range(n), :] = 0   # diagonal 0
    qi = qt.sum(axis=1)             # rowsum
    qt = qt / qi[:, None, :]        # normalize
    z = np.where(qi == 0)           # special case q_i = 0 => q_ij = kron_ij
    qt[z[0], :,    z[1]] = 0
    qt[z[0], z[0], z[1]] = 1
    return qt, qi


def test_singlejump():
    p = BirthDeath()
    p.ajc.jump(p.p_0)


def run(p=BirthDeath(n_t=100, L=1), n_iter=200):
    x = p.p_0
    xs = [x]
    # p_X_T = p_0.copy()

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

    # p = p_0
    for i in range(n_iter):

        x = p.ajc.jump(x)
        xs.append(x)
        # p_X_T += p

        print(np.sum(x))

        pl.clf()
        pl.title('step:' + str(i))
        pl.imshow(x.T, origin='lower', aspect='auto')
        # pl.imshow(p_X_T.T, origin = 'low', aspect = 'auto')
        # inds = range(len(T))[::40]
        # pl.yticks(inds, T[inds])
        pl.xlabel('X')
        pl.ylabel('Time')
        pl.colorbar()

        pl.draw()
        pl.pause(0.01)

    return xs


if __name__ == '__main__':
    run()
    pdb.set_trace()
