from ajcs import AJCS
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SqraOptim:
    def __init__(self, sqra, ts, adaptive = False, x0 = None, penalty=1):
        j = AJCS([sqra.Q] * len(ts), ts)
        if x0 is None:
            x = np.zeros_like(sqra.u.flatten())
        else:
            x = x0
        iters = 0
        hist_x = np.empty((0,len(x)))
        hist_o = []

        savevars(self, locals())  # think about this, not really pythonic, but so comfy...

        self.objective(x)
        
    
    def optimize(self, iters=100):
        def obj(x):
            return self.objective(x)
        
       
        res = minimize(obj, x0=self.x, method='nelder-mead', options={'maxiter': iters, 'adaptive': self.adaptive})
        self.objective(res.x)
        return res

    def objective(self, x):
        j = self.j
        j.Q = [self.sqra.perturbed(x)]*len(j.dt)
        j.compute()

        o = - np.min(j.finite_time_hitting_probs())
        o += np.sum(abs(x)) * self.penalty

        #print(o)

        self.iters += 1 
        self.x = x
        self.o = o
        self.hist_x = np.vstack((self.hist_x, x))
        self.hist_o = np.append(self.hist_o, o)
        
        return o
    
    def plot(self):
        o = self
        fig, ax = plt.subplots(3,1, constrained_layout=True, figsize=(6,8))
        fig.suptitle("penalty: " + str(self.penalty))
        
        #plt.figure(figsize=(8,12))
        #ax1 = plt.subplot(3,1,1)
        im = ax[0].imshow(o.hist_x.T, aspect='auto', interpolation='nearest')
        ax[0].set_title("potential history")
        plt.colorbar(im, ax=ax[0])
        
        ax[1].plot(o.hist_o)
        penalty = np.sum(o.hist_x, axis=1) * self.penalty
        #ax[1].plot(o.hist_o - penalty)
        #ax[1].plot(penalty)
        ax[1].set_title("objective history")

        ax[2].set_title("initial and optimal potential")
        u = o.sqra.u.flatten()
        ax[2].plot(u)
        ax[2].plot(u + o.x)
 

def penaltyheuristic(sqra):
    return 1 / np.sum(np.max(sqra.u)-sqra.u)



def savevars(obj, vardict):
    for key, value in vardict.items():
        setattr(obj, key, value)