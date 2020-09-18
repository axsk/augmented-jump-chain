from ajcs import AJCS
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SqraOptim:
    def __init__(self, sqra, ts, adaptive = False, x0 = None, penalty=1):
        self.sqra = sqra
        self.penalty = penalty
        self.adaptive = adaptive

        self.j = AJCS([sqra.Q] * len(ts), ts)
        self.x = self.default_x0() if x0 is None else x0
        self.iters = 0
        self.hist_x = np.empty((0,len(self.x)))
        self.hist_x = []
        self.hist_o = []
        self.simplex = None

        self.objective(self.x)
    
    def default_x0(self):
        return np.zeros(self.j.nx)
    
    def optimize(self, iters=100):
        def obj(x):
            return self.objective(x)

        res = minimize(obj, x0=self.x, method='nelder-mead', options={'maxiter': iters, 'adaptive': self.adaptive, 'initial_simplex':self.simplex})
        self.objective(res.x)
        self.simplex = res.final_simplex[0]
        return res

    def perturb(self, x):
        j = self.j
        j.Q = [self.sqra.perturbed(x)]*len(j.dt)
        j.compute()

    def objective(self, x):
        self.perturb(x)

        hp = self.j.finite_time_hitting_probs()
        hp = hp[hp.nonzero()]
        o = - np.min(hp)
        o += np.sum(abs(x)) * self.penalty

        self.iters += 1 
        self.x = x
        self.o = o
        self.hist_x.append(x.flatten())
        self.hist_o.append(o)
        
        return o
    
    def plot(self):
        o = self
        fig, ax = plt.subplots(3,1, constrained_layout=True, figsize=(6,8))
        fig.suptitle("penalty: " + str(self.penalty))
        
        #plt.figure(figsize=(8,12))
        #ax1 = plt.subplot(3,1,1)
        histx = np.reshape(np.array(o.hist_x), (len(o.hist_o), -1))
        im = ax[0].imshow(histx.T, aspect='auto', interpolation='nearest')
        ax[0].set_title("perturbation history")
        plt.colorbar(im, ax=ax[0])
        
        ax[1].plot(o.hist_o)
        penalty = np.sum(o.hist_x, axis=1) * self.penalty
        ax[1].plot(o.hist_o - penalty)
        #ax[1].plot(penalty)
        ax[1].set_title("objective history")

        self.plot_initial_optimal_potential(ax[2])



    def plot_initial_optimal_potential(self, ax):
        ax.set_title("initial and optimal potential")
        u = self.sqra.u.flatten()
        ax.plot(u)
        ax.plot(self.x+u)

    def plot_fin_hit_prob(self):
        plt.imshow(self.j.finite_time_hitting_probs())
        plt.colorbar()
        plt.title("fin. time hitting prob.")
        plt.ylabel("start")
        plt.xlabel("hit")


class SqraOptimNonaut(SqraOptim):
    def default_x0(self):
        return np.zeros((self.j.nt, self.j.nx))

    def perturb(self, x):
        j = self.j
        xts = np.reshape(x, (j.nt, j.nx))
        j.Q = [self.sqra.perturbed(xt) for xt in xts] 
        j.compute()
    
    def plot_initial_optimal_potential(self, ax):
        # ax.set_title("initial and optimal potential")
        # u = self.sqra.u.flatten()
        # u = np.tile(u, self.j.nt)
        # ax.plot(u)
        # ax.plot(self.x+u)

        x = np.reshape(self.x, (self.j.nt, self.j.nx))
        u = self.sqra.u.flatten()
        im = ax.imshow(x+u)
        plt.colorbar(im, ax=ax)
        plt.xlabel("x")
        plt.ylabel("t")
        ax.set_title("optimal potential")

    
    def scatter(self):
        x = self.x.reshape(self.j.nt, self.sqra.ny, self.sqra.nx)
        return scatter_3d_array(x)
 
import plotly.express as px

def scatter_3d_array(x):
    npoints = np.size(x)
    z = np.ones((npoints, 5))
    for l in range(npoints):
        i,j,k = np.unravel_index(l, x.shape)
        val = x[i,j,k]
        z[l,0:3] = i,j,k
        z[l,3] = np.abs(val)
        z[l,4] = val
    
    return px.scatter_3d(z, x=0, y=1, z=2, size=3, color=4,  labels = {'0':'t', '1':'y', '2':'x'})


def penaltyheuristic(sqra):
    return 1 / np.sum(np.max(sqra.u)-sqra.u)



def savevars(obj, vardict):
    for key, value in vardict.items():
        setattr(obj, key, value)