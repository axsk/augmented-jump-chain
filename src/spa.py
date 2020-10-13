"""
Scalable Probabilistic Approximation
as in
[1] https://advances.sciencemag.org/content/6/5/eaaw0961
[2] https://advances.sciencemag.org/content/advances/suppl/2020/01/27/6.5.eaaw0961.DC1/aaw0961_SM.pdf

eqn. numbers (123) refer to [2]
"""

import numpy as np
import quadprog

class Spa_core:
    """
    SPA for least-squares approximation with tikhonov regularization (SPA2 in [2] p.16)
    [S, T] = argmin ||X-ST||_2 + eps * H(S)
    where X is the data, S the cluster- and T the probability-matrix.
    H denotes a (modified) Tikhonov regularization term.
    """

    @staticmethod
    def iterate(X, G, eps):
        """ Coordinate descent as in (Algorithm 1) """
        S = Spa_core.solve_S(X, G, eps)
        G = Spa_core.solve_G(X, S)
        return S, G

    @staticmethod
    def solve_S(X, G, eps, use_inverse=False):
        """ solve S in euclidean case (spa2) with tikh regularization (27)
        due to Lemma (28) """
        n = np.shape(X)[0]
        K = np.shape(G)[0]

        h = 2 * eps**2 / (n * K * (K-1))
        H = G @ G.T + h * (K * np.identity(K) - 1)
        if use_inverse:
            S = X @ G.T @ np.linalg.inv(H)
        else:
            S = np.linalg.solve(H.T, G @ X.T).T

        return S

    @staticmethod
    def solve_G(X, S):
        """ solve T via the quadratic progamm from Lemma 11 (34)  """
        K = np.size(S, 1)
        T = np.size(X, 1)
        G = np.empty((K, T))

        A = S.T @ S

        # eq. and ineq. constraints
        O = np.ones((1, K))
        I = np.identity(K)
        C = np.vstack([O, I]).T
        c = np.hstack([1, np.zeros(K)])

        for t in range(T):
            b = S.T @ X[:, t]
            res = quadprog.solve_qp(A, b, C, c, 1)
            G[:, t] = res[0]

        return G


class Spa(Spa_core):
    """
    convenience wrapper around the Spa_core class,
    offering an optimizer with convergence criteria
    """

    def __init__(self, X, k=2, eps=1e-3, maxiter=1000, reltol=1e-6, abstol=1e-12, verbose=True, skipobjective=False, G=None):
        self.X = X
        self.k = k
        self.eps = eps
        self.verbose = verbose
        self.skipobjective = skipobjective
        self.reltol = reltol
        self.abstol = abstol
        self.maxiter = maxiter
        self.core = Spa_core

        self.n, self.T = np.shape(X)
        self.niter = 0

        # initial value for optimization
        self.S = np.zeros((self.n, self.k))
        self.G = self.random_G() if G is None else G

    def random_G(self):
        # generate random assignment matrix
        G = np.zeros([self.k, self.T])
        inds = np.random.choice(self.k, self.T)
        G[inds, range(self.T)] = 1
        return G / np.sum(G, axis=0)

    def solve(self):
        LL = self.objective()

        for i in range(self.maxiter):
            self.S, self.G = self.core.iterate(self.X, self.G, self.eps)
            self.niter += 1

            L = self.objective()
            if np.abs(L - LL) < (self.reltol * np.abs(LL) + self.abstol):
                break
            LL = L

        return self

    def objective(self):
        """
        objective function for euclidean spa with tikh regularization.
        note the unusual pairwise comparison in the tikh term.
        this function can be slow in comparison to the iteration itself.
        """

        if self.skipobjective:
            return np.inf

        error = np.linalg.norm(self.X - self.S @ self.G)**2

        tikh  = 0
        for k1 in range(self.k):
            for k2 in range(self.k):
                tikh += np.sum(np.square(self.S[:,k1] - self.S[:,k2]))
        tikh *= self.eps ** 2 / (self.n * self.k * (self.k - 1))

        obj = error + tikh

        if self.verbose:
            print(f"objective: {obj:.6f} approx-error: {error:.6f}")

        return obj



from matplotlib import pyplot as plt

def test_spa(n=5, T=100):
    X = test_data(T, n)
    spa = Spa(X, k=4, skipobjective=True, verbose=True)
    spa.solve()
    return spa

def test_data(T=100, n=3):
    # T draws from a mixture of 4 n-dimensional gaussians
    k = 4
    mu = np.zeros((k, n))
    mu[1, 0:2] = [.8, 1.6]
    mu[2, 0] = 1.6
    mu[3, 0:2] = 0.8

    sig = np.zeros((k, n, n))
    sig[0, 0:2, 0:2] = [[.1, .05],[.05, .1]]
    sig[1, 0:2, 0:2] = [[.1, -.05],[-.05, .1]]
    sig[2, 0:2, 0:2] = np.identity(2)
    sig[:, 2:n, 2:n] = np.identity(n-2) * .2/(n-2)
    sig[3] = sig[2]

    inds = [t%k for t in range(T)]
    inds = np.sort(inds)
    X = np.array([np.random.multivariate_normal(mu[i,:], sig[i,:,:]) for i in inds])

    return X.T
