import numpy as np
from numpy.core.numeric import identity


class AJC:
    def jump(self, p):
        return np.tensordot(self.k, p, ([0, 1], [0, 1]))  # sum over first two inds

    def jump2(self, p):
        raise NotImplementedError
        # the following is wrong assuming wrong index orders
        nx, nt = p.shape
        return self.km.dot(p.reshape(nx*nt)).reshape(nx, nt)

    def jump3(self, p):
        return np.einsum('isjt, is -> jt', self.k, p)

    def geiger(self, p, n=100):
        g = np.copy(p)
        pp = p
        for i in range(n):
            pp = self.jump(pp)
            g = g + pp
        return g

    def holding_probs(self):
        S = self.k.sum(axis=2)  # sum over target points
        S = 1 - np.cumsum(S, axis=2)  # cumsum to get prob to jump in interval
        S = np.triu(S)
        return S

    def synchronize(self, g):
        S = self.holding_probs()
        return np.einsum('ist, is -> it', S, g)

    def commitor(self, g, inds_bnd: 'np.ndarray[bool]'):

        km = self.km
        inds_inner = ~inds_bnd

        km_ii = km[np.ix_(inds_inner, inds_inner)]  # inner
        km_ib = km[np.ix_(inds_inner, inds_bnd)]  # boundary

        ni = np.size(km_ii, 0)
        inv = np.linalg.inv(np.identity(ni) - km_ii)

        S = self.holding_probs()
        Sg = np.einsum('is, i -> is', S[:, :, -2], g).T.flatten()[inds_inner]

        q_i = inv.dot(km_ib.dot(g) + Sg)

        q = np.zeros(self.km.shape[0])
        q[inds_inner] = q_i
        q[inds_bnd] = g

        return q

    def finite_time_hitting_prob(self, n_state):
        """ Compute the probability to hit a given state n_state over the
        time horizon of the jump chain from any space-time point by solving
        Kp = p in C, p=1 in C'
        where C' is the time-fibre of n_state and C the rest """
        nx, nt = self.k.shape[0:2]
        km = self.km

        # Kp - p = 0 in C
        M = km - np.identity(nx*nt)
        b = np.zeros(nx*nt)

        # p = 1 in C'
        for i in range(nt):
            st = i*nx + n_state  # spacetime index for state n_state at time i
            M[st, :] = 0
            M[st, st] = 1
            b[st] = 1

        p = np.linalg.inv(M).dot(b)
        return p





    def unflatten(self, x, dims=None):
        if dims is None:
            dims = self.k.shape[0:2]

        if len(x) == np.prod(dims):
            return np.reshape(x, np.flip(dims)).T
        elif len(x) == np.prod(dims)**2:
            return kernel_to_matrix(x)
        else:
            raise Exception()


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


def kernel_to_matrix(k):
    """ given k[i,s,j,t] return the row-stochastic, space-major
    transition matrix M """
    nxt = np.shape(k)[0] * np.shape(k)[1]
    return k.T.reshape(nxt, nxt).T


def holding_probs(qi, dt):
    """ compute the holding probabilities from outbound rates assuming
    they are constant along the time intervals.
    input: qi[x,s]:   outbound rate at (x,s),
        dt[s]:     length of timeintervals :math:`\\Delta T_s``$
    output: S[x,s,t]: probability to stay in x from s to t
    """
    nt = len(dt)
    assert np.size(qi, 1) == nt

    dts = np.zeros((nt, nt))
    for i in range(nt):
        dts[i, i:] = np.cumsum(dt[i:])

    # S[i,s,t] = exp(-(t-s) * qi[s])
    S = np.exp(-np.einsum('xs, st -> xst', qi, dts))
    S = np.triu(S)
    return S


class AJCGalerkin(AJC):
    def __init__(self, Q, dt):
        self.Q = Q
        self.dt = dt
        self.qt, self.qi = qtilde(Q)
        self.k, self.H = self.jumpkernel(self.qt, self.qi, self.dt)
        self.km = kernel_to_matrix(self.k)

    @staticmethod
    def jumpkernel(qt, qi, dt):
        """ compute the galerkin discretization of the jump kernel eq[50] """

        s = np.exp(-np.einsum('ik, k -> ik', qi, dt))
        nx, nt = np.shape(qi)

        H = np.zeros((nx, nt, nt))
        for i in range(nx):
            for k in range(nt):
                prod = np.insert(np.cumprod(s[i, k+1:-1]), 0, 1)
                H[i, k, k+1:] = (1-s[i, k]) * (1-s[i, k+1:]) * prod
                H[i, k, k] = s[i, k] + dt[k] * qi[i, k] - 1

        J = np.einsum('k, ijl, ik, ikl -> ikjl', 1/dt, qt, 1/qi, H)
        return J, H


class AJCCollocation(AJC):
    def __init__(self, Q, S):
        self.Q = Q
        self.qt, self.qi = qtilde(Q)
        self.S = S
        self.k = self.jumpkernel(self.qt, self.qi, self.S)
        self.km = kernel_to_matrix(self.k)

    @staticmethod
    def jumpkernel(qt, qi, S):
        """
        compute the jumpkernel for given integrals S[i,s,t] = exp(-int_s^t q_i)
        this is the collocation approach, i.e. take the density of k(i,s,j,t) as a
        transtion probability (i,s) -> (j,t).
        """
        k = np.einsum('ijt, it, ist -> isjt', qt, qi, S)
        k = k / k.sum(axis=(2, 3))[:, :, None, None]  # normalize density to probability
        return k
