import numpy as np


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

    # def synchronize
    # def koop

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
        self.k = jumpkernel_coll(self.qt, self.qi, self.S)
        self.km = kernel_to_matrix(self.k)

    @staticmethod
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








