import ajc
import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import spsolve
from copy import deepcopy
from scipy.optimize import minimize


class AJCS(ajc.AJCGalerkin):
    def __init__(self, Q, dt):
        self.nx = Q[0].shape[0]
        self.nt = len(dt)
        self.nxt = self.nx * self.nt
        self.Q = Q
        self.dt = np.array(dt)
        self.compute()

    def compute(self):
        self.qt, self.qi = self.qtilde(self.Q)
        self.k, self.H, self.S = self.jumpkernel(self.qt, self.qi, self.dt)

    @staticmethod
    def jumpkernel(qt, qi, dt):
        """ compute the galerkin discretization of the jump kernel eq[50] """

        nx, nt = np.shape(qi)
        s = np.exp(-qi * dt[None, :])  # (nx, nt)

        H = np.zeros((nx, nt, nt))
        for i in range(nx):
            for k in range(nt):
                prod = np.append(1, np.cumprod(s[i, k+1:-1]))
                H[i, k, k+1:] = (1-s[i, k]) * (1-s[i, k+1:]) * prod
                H[i, k, k] = s[i, k] + dt[k] * qi[i, k] - 1

        #J = np.einsum('k, ijl, ik, ikl -> ikjl', 1/dt, qt, 1/qi, H)
        J = np.empty((nt, nt), dtype=object)
        S = np.zeros((nt, nt, nx))
        for k in range(nt):
            for l in range(nt):
                S[k, l, :] = (1/dt[k]) * H[:, k, l] / (qi[:, k])
                J[k, l] = sp.diags(S[k, l, :]).dot(qt[l])
                # is  csr_scale_rows faster?
        return J, H, S

    def finite_time_hitting_prob(self, state):
        """ Compute the probability to hit a given state n_state over the
        time horizon of the jump chain from any space-time point by solving
        Kp = p in C, p=1 in C'
        where C' is the time-fibre of n_state and C the rest """
        nx, nt = self.nx, self.nt
        K = deepcopy(self.k)

        b = np.zeros((nt, nx))

        # Kp - p = 0 in C
        for s in range(nt):
            for t in np.arange(s, nt):
                K[s, t][state, :] = 0

            K[s, s] = K[s, s] - sp.identity(nx)
            K[s, s][state, state] = 1
            b[s, state] = 1

        q = self.backwardsolve(K, b)
        return q

    def backwardsolve(self, K, b):
        nt, nx = np.shape(b)
        q = np.zeros((nt, nx))

        for s in range(nt)[::-1]:
            for t in np.arange(s, nt)[::-1]:
                if s < t:
                    b[s] -= K[s, t].dot(q[t])
                elif s == t:
                    q[s] = spsolve(K[s, s], b[s])

        return q

    def finite_time_hitting_probs(self):
        """ finite_time_hitting_probs[i,j] is the probability to hit state j starting in i in the time window of the process """
        # TODO: check if this is indeed the ordering
        return np.vstack([self.finite_time_hitting_prob(i)[0,:] for i in range(self.nx)]).T

    def optimize(self, iters=100, penalty=0, x0=None, adaptive=False):
        # TODO: assert identical sparsity structures
        j = self
        jp = deepcopy(self)
        if x0 is None:
            x0 = np.zeros_like(j.Q[0].data)
        
        def obj(x):
            self.lastx = x
            for t in range(j.nt):
                #jp.Q[t].data = np.maximum(j.Q[t].data + x, 0)  # TODO: ignore diagonal
                jp.Q[t].data = j.Q[t].data + x
            jp.compute()
            o = - min(jp.finite_time_hitting_probs())
            op = o + np.sum(np.abs(x)) ** 2 * penalty 
            print(op)
            return op

            # q = sqrt (exp -bU / exp -bU)
        
        res = minimize(obj, x0=x0, method='nelder-mead', options={'maxiter': iters, 'adaptive': adaptive})
        obj(res.x)
        return res, jp

    @staticmethod
    def qtilde(Qs):
        """ given a standard rate matrix returns:
        qt[i,j,t] = q^tilde_ij(t) : the row normalized jump matrix [eq. 14]
        qi[i,t] = q_i(t): the outflow rate [eq. 6]
        """

        qt = [sp.dok_matrix(Q) for Q in Qs]
        nt = len(qt)
        nx = qt[0].shape[0]
        qi = np.zeros((nt, nx))
        for k in range(nt):
            qt[k][range(nx), range(nx)] = 0
            qi[k, :] = qt[k].sum(axis=1).A[:, 0]
            qt[k] = sp.diags(1/qi[k, :]).dot(qt[k])
            z = np.where(qi[k, :] == 0)           # special case q_i = 0 => q_ij = kron_ij
            qt[k][z[0], :] = 0
            qt[k][z[0], z[0]] = 1

        return qt, qi.T

 

def flatten_spacetime(tensor):
    ns, nt = tensor.shape
    for s in range(ns):
        for t in range(nt):
            if t == 0:
                row = tensor[s, t]
            else:
                row = sp.hstack((row, tensor[s, t]))
        if s == 0:
            M = row
        else:
            M = sp.vstack((M, row))
    return M





