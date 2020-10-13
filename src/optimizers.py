import numpy as np

class Optimizer:
    def run(self, iter):
        for i in range(iter):
            self.iterate()

    def iterate(self):
        pass

    def initialize(self, f, df, x0):
        self.call_f = f
        self.call_df = df
        self.x = x0
        self.reset()

    def reset(self):
        pass

class Rprop(Optimizer):
    def __init__(self, g0=None, inc=1.2, dec=.5, max=50, min=0):
        self.g = g0
        self.inc = inc
        self.dec = dec
        self.max = max
        self.min = min

    def reset(self):
        n = len(self.x)
        self.lastdf = np.zeros(n)
        if self.g is None:
            self.g = np.ones(n)
        self.g = self.g / self.dec # since we will decrease on first run

    def iterate(self):
        df = self.call_df(self.x)
        g = self.g
        for i in range(len(g)):
            if df[i] * self.lastdf[i] > 0:
                g[i] = np.minimum(g[i] * self.inc, self.max)
            else:
                g[i] = np.maximum(g[i] * self.dec, self.min)

        dx = -g * np.sign(df)
        self.lastdf = df
        self.x += dx

class Momentum(Optimizer):
    def __init__(self, alpha=0.5, h=1):
        self.alpha = alpha
        self.h = h

    def reset(self):
        self.dx = np.zeros_like(self.x)

    def iterate(self):
        self.df = self.call_df(self.x)
        self.dx =  self.alpha * self.dx - self.h * self.df
        self.x += self.dx

class RMSProp(Optimizer):
    def __init__(self, h=1, gamma=0.9):
        self.h = h
        self.gamma = gamma

    def reset(self):
        self.v = np.zeros_like(self.x)

    def iterate(self):
        self.df = self.call_df(self.x)
        self.v = self.gamma * self.v + (1 - self.gamma) * (self.df ** 2)
        self.x -= self.h * 1 / np.sqrt(self.v) * self.df

class Adam(Optimizer):
    def __init__(self, h=1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.h = h
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def reset(self):
        self.n = 0
        self.m = np.zeros_like(self.x)
        self.v = np.zeros_like(self.x)

    def iterate(self):
        self.df = self.call_df(self.x)
        self.n += 1

        self.m = self.beta1 * self.m + (1-self.beta1) * self.df
        self.v = self.beta2 * self.v + (1-self.beta2) * (self.df ** 2)
        self.mh = self.m / (1 - self.beta1 ** self.n)
        self.vh = self.v / (1 - self.beta2 ** self.n)

        self.x -= self.h * self.mh / (np.sqrt(self.vh) + self.eps)


from scipy.optimize import minimize

class ScipyOpt(Optimizer):
    def __init__(self, **kwargs):
        self.args = kwargs

    # TODO: dont ignore iter
    def run(self, iter):
        self.x = minimize(self.call_f, self.x, jac=self.call_df, **self.args, options={"maxfev":iter}).x