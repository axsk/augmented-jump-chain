import ajc
import scipy.sparse as sp

class AJCS(ajc.AJCGalerkin):
    def __init__(self, Q, dt):
        self.nx = np.size(Q, axis=0)
        self.nt = len(dt)
        self.nxt = self.nx * self.nt
        self.Q = Q
        self.dt = dt
        self.qt, self.qi = qtilde(Q)
        self.k, self.H = self.jumpkernel(self.qt, self.qi, self.dt)
        #self.km = kernel_to_matrix(self.k)

    @staticmethod
    def jumpkernel(qt, qi, dt):
        """ compute the galerkin discretization of the jump kernel eq[50] """

        nt, nx = np.shape(qi)
        s = np.exp(-qi * dt[None, :])

        H = np.zeros((nx, nt, nt))
        for i in range(nx):
            for k in range(nt):
                prod = np.insert(np.cumprod(s[k+1:-1,i]), 0, 1)
                H[i, k, k+1:] = (1-s[i, k]) * (1-s[i, k+1:]) * prod
                H[i, k, k] = s[i, k] + dt[k] * qi[i, k] - 1

        #J = np.einsum('k, ijl, ik, ikl -> ikjl', 1/dt, qt, 1/qi, H)
        J = np.empty((nt, nt), dtype=object)
        for k in range(nt):
            for l in range(nt):
                J[k,l] = 1/dt[k] * sp.diags(1 / (qi[:, k] * H[:, k, l])).dot(qt[l])
                # is  csr_scale_rows faster?
        return J, H

from  copy import deepcopy

import numpy as np

def qtilde(Qs):
    """ given a standard rate matrix returns:
    qt[i,j,t] = q^tilde_ij(t) : the row normalized jump matrix [eq. 14]
    qi[i,t] = q_i(t): the outflow rate [eq. 6]
    """

    qt = [sp.dok_matrix(Q) for Q in Qs]
    nt = len(qt)
    nx = qt[0].shape[0]
    qi = np.zeros((nt,nx))
    for k in range(nt):
        qt[k][range(nx), range(nx)] = 0
        qi[k, :] = qt[k].sum(axis=1).A[:,0]
        qt[k] = sp.diags(1/qi[k, :]).dot(qt[k])
        z = np.where(qi[k,:] == 0)           # special case q_i = 0 => q_ij = kron_ij
        qt[k][z[0], :] = 0
        qt[k][z[0], z[0]] = 1

    return qt, qi.T


