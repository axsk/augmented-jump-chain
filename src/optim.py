import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import ode

class SqraOptim:
    def __init__(self, sqra, dts, penalty=1, x0=None, adaptive=False, nonautonomous=True):
        self.sqra = sqra
        self.dts = dts
        self.penalty = penalty
        self.adaptive = adaptive
        self.nonautonomous = nonautonomous

        self.nt = len(dts)
        self.nx = sqra.N

        self.x = self.default_x0() if x0 is None else x0

        self.reset()


    ### Optimizer

    def default_x0(self):
        if self.nonautonomous:
            return np.zeros((self.nt, self.nx))
        else:
            return np.zeros(self.nx)

    def optimize(self, iters=100):
        def obj(x):
            o = self.objective(x) + self.penaltyfunction(x)
            self.saveiter(x, o)
            return o

        res = minimize(obj, x0=self.x, method='nelder-mead', options={'maxiter': iters, 'adaptive': self.adaptive, 'initial_simplex':self.simplex})
        self.simplex = res.final_simplex[0]
        self.x = res.x
        obj(res.x)
        return res

    def reset(self):
        self.iters = 0
        self.hist_x = []
        self.hist_o = []
        self.simplex = None

    def saveiter(self, x, o):
        self.iters += 1
        self.x = x
        self.o = o
        self.hist_x.append(x.flatten())
        self.hist_o.append(o)


    ### Objective

    def objective(self, x):
        hp = self.finite_time_hitting_probs(x)
        hp = hp[hp.nonzero()]  # ignore fields which are not hit at all (infinity potential)
        return -np.min(hp)     # maximze the minimal hitting prob

    def finite_time_hitting_probs(self, x) -> np.ndarray:
        Qs = self.perturbed_Qs(x)
        hp = ode.finite_time_hitting_probs(Qs, self.dts) # we worked with the ajc before, but this is faster
        return hp

    def perturbed_Qs(self, x):
        if self.nonautonomous:
            xs = np.reshape(x, (self.nt, self.nx))
            return [self.sqra.perturbed(x) for x in xs]
        else:
            return [self.sqra.perturbed(x)] * self.nt

    def penaltyfunction(self, x):
        return np.sum(np.abs(x)) * self.penalty


    ## Plot functions

    def plot(self):
        fig, ax = plt.subplots(3,1, constrained_layout=True, figsize=(6,8))
        fig.suptitle("penalty: " + str(self.penalty))

        self.plot_perturbation_history(ax[0])
        self.plot_objective_history(ax[1])
        self.plot_initial_optimal_potential(ax[2])

    def plot_perturbation_history(self, ax):
        histx = np.reshape(np.array(self.hist_x), (len(self.hist_o), -1))
        im = ax.imshow(histx.T, aspect='auto', interpolation='nearest')
        ax.set_title("perturbation history")
        plt.colorbar(im, ax=ax)

    def plot_objective_history(self, ax):
        ax.plot(self.hist_o)
        penalty = np.sum(self.hist_x, axis=1) * self.penalty
        ax.plot(self.hist_o - penalty)
        ax.set_title("objective history")

    def plot_initial_optimal_potential(self, ax):
        if self.nonautonomous:
            x = np.reshape(self.x, (self.nt, self.nx))
            u = self.sqra.u.flatten()
            im = ax.imshow(x+u)
            plt.colorbar(im, ax=ax)
            plt.xlabel("x")
            plt.ylabel("t")
            ax.set_title("optimal potential")
        else:
            u = self.sqra.u.flatten()
            ax.plot(u)
            ax.plot(self.x+u)
            ax.set_title("initial and optimal potential")

    def plot_fin_hit_prob(self):
        plt.imshow(self.finite_time_hitting_probs(self.x))
        plt.colorbar()
        plt.title("fin. time hitting prob.")
        plt.ylabel("start")
        plt.xlabel("hit")

    def plot_perturbation(self):
        x = self.x.reshape(self.nt, self.nx)
        plt.imshow(x)
        plt.colorbar()

    def scatter(self):
        x = self.x.reshape(self.nt, self.sqra.ny, self.sqra.nx)
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

import plotly.graph_objects as go

def stack_3darray(x):
    return go.Figure(data=[go.Surface(z = z) for z in x])


def penaltyheuristic(sqra):
    return 1 / np.sum(np.max(sqra.u)-sqra.u)


def savevars(obj, vardict):
    for key, value in vardict.items():
        setattr(obj, key, value)