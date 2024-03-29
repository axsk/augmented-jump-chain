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
from matplotlib import pyplot as plt
from softmin import fd_softmin_rel


class hitting_prob_adjoint:
    """
    Class for computing the finite time hitting probability
    for given SQRA at different times.
    Also computes the derivative of the desired hitting probability wrt. to the potential u
    """

    def __init__(self, sqras, dts, nquad=100):
        self.dts = dts
        self.cts = np.cumsum(dts)

        self.nx = sqras[0].N
        self.nt = len(dts)

        self.nquad = nquad # integration stepsize
        self.its = np.linspace(0, self.cts[-1], nquad)

        #self.sqras = sqras
        self.Qs = [s.Q for s in sqras]
        self.us = [s.u for s in sqras]
        self.dQs = [s.dQ for s in sqras]

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

    def ode_mu(self, i, j):
        # adjoint ode for the derivative of the active hp

        def dmu(mu, t):
            mu = mu.copy()
            mu[j] = 0  # equivalent to setting the j-th row of Q to 0
            d = -self.dfdy(t).T.dot(mu)
            d = np.nan_to_num(d)
            return d

        mu0 = np.zeros(self.nx)
        mu0[i] = 1

        mu = odeint(dmu, mu0, self.its)
        return mu

    def dfdp(self, t, y, j):
        # df/dp = -dQ/dp * y
        dfdp = self.dQs[self.timeindex(t)](y)
        dfdp[j] = 0
        return dfdp

    def adjointintegrate(self, mu, y, j):
        # int_0^T mu^* f_p
        its = self.its
        dg = np.zeros(self.nx)

        for i in range(len(its)-1):
            dt = its[i+1] - its[i]
            dqy = self.dfdp(its[i], y[i], j)
            dg += dt * dqy.T.dot(mu[i])

        return dg

    def hitting_probs(self):
        # compute all hitting probs and save the active minimium
        # hps[i,j] is the prob. to hit j when starting in i
        hps = np.zeros((self.nx, self.nx))

        for j in range(self.nx):
            y = self.ode_y(j)
            hps[:,j] = y[0,:]

        self.hps = hps
        return hps

    def derivative(self, i, j):
        y = self.ode_y(j)
        mu = self.ode_mu(i, j)
        dg = self.adjointintegrate(mu, y, j)

        return dg


class hitting_prob_min:
    """ using the hitting_prob_adjoint solver,
    compute the (relaxed) minimal hitting probabilities
    """
    def __init__(self, hitting_prob_adjoint, maxsubder=1, rtol=np.inf, softminscale = 30):
        self.hpa = hitting_prob_adjoint
        self.nx = self.hpa.nx
        self.maxsubder = maxsubder
        self.rtol = rtol
        self.softminscale = softminscale

    def relaxedmin(self):
        hps = self.hpa.hitting_probs()
        inds = self.sorthps(hps)

        dg = np.zeros((self.nx))
        hpmin = hps[inds[0,0],inds[0,1]]

        hpm = []
        dgs = []

        for ind in inds[0:self.maxsubder]:
            i,j = ind
            hp = hps[i,j]
            if hp > hpmin * (1+self.rtol):
                break

            hpm.append(hp)
            dgs.append(self.hpa.derivative(i,j))

        sm, dsm = fd_softmin_rel(np.array(hpm), self.softminscale)

        hp = sm
        dg = dsm @ dgs

        return hp, dg

    def min(self):
        hps = self.hpa.hitting_probs()
        i, j  = self.sorthps(hps)[0,:]
        dg = self.hpa.derivative(i, j)
        return hps[i,j], dg

    def sorthps(self, hps):
        """ given the hps matrix, return the indices of the minima
        inds[i,1] -> inds[i,2] is the i-th lowest hitting probability """
        inds = np.unravel_index(np.argsort(hps, axis=None), shape=(self.nx, self.nx))
        inds = np.vstack(inds).T
        return inds


from optimizers import Rprop

class HittingProbOptimization:
    def __init__(self, sqra, T, penalty=0.00001, x0=None, maxsubder=1, optimizer=Rprop(), verbose=False, softminscale=30):
        self.sqra = sqra
        self.T = T
        self.penalty = penalty
        self.maxsubder = maxsubder
        self.verbose = verbose
        self.softminscale = softminscale

        if x0 is None:
            x0 = np.zeros(self.sqra.N)

        self.x = None
        self.histx = []
        self.histobj = []
        self.histmin = []
        self.histdobj = []
        self.histdmin = []
        self.histdcost = []

        self.perturb(x0)

        self.optim = optimizer
        self.optim.initialize(f=self.objcall, df=self.dobjcall, x0=self.x)

    def describe(self):
        return self.optim.__class__.__name__ + " sd" + str(self.maxsubder) + "sc " + str(self.softminscale)

    def objcall(self, x):
        self.perturb(x)
        return -self.obj

    def dobjcall(self, x):
        self.perturb(x)
        return -self.dobj

    def print_status(self):
        print(f"hp:{self.hpmin}, cost:{self.cost}, obj:{self.obj}")

    def perturb(self, x):
        self.x = x.copy()
        self.sqra_perturbed = self.sqra.perturbed(x)
        self.hpa = hitting_prob_adjoint([self.sqra_perturbed], [self.T])
        self.hpm = hitting_prob_min(self.hpa, self.maxsubder, softminscale=self.softminscale)

        self.hpmin, self.dhpmin = self.hpm.relaxedmin()

        self.cost = self.penalty * np.sum(np.abs(x))
        self.dcost = self.penalty * np.sign(x)

        self.obj  = self.hpmin  - self.cost
        self.dobj = self.dhpmin - self.dcost


        self.histx.append(self.x)
        self.histobj.append(self.obj)
        self.histmin.append(self.hpmin)
        self.histdobj.append(self.dobj)
        self.histdcost.append(self.dcost)
        self.histdmin.append(self.dhpmin)

        if self.verbose:
            self.print_status()

    def plot_gradient(self, penalty=True):
        g = self.dhpmin
        if penalty:
            g = self.dobj

        self.imshow(g, center=True)

    def plot_x(self):
        plt.title(f"hp:{self.hpmin:.4} cost:{self.cost:.4}")
        self.imshow(self.x, center=True)

    def plot_g(self):
        self.imshow(self.optim.g)

    def plot_u(self):
        self.imshow(self.s.u)

    def plot_xhist(self):
        plt.plot(np.array(self.histx))
        plt.legend(range(len(self.x)))
        plt.show()

    def plot_objhist(self):
        plt.plot(np.array(self.histobj))
        plt.plot(np.array(self.histmin))
        plt.show()

    def plot_derivatives(self):
        plt.plot(self.histdobj)
        plt.legend(range(len(self.x)))
        plt.show()

    def imshow(self, x, center=False):
        if center:
            m = np.max(np.abs(x))
            vmin, vmax = -m, m
            cmap = cmap=plt.get_cmap('bwr')
        else:
            vmin, vmax = np.min(x), np.max(x)
            cmap = cmap=plt.get_cmap('cividis')

        plt.imshow(np.reshape(x, self.sqra.dims), cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()