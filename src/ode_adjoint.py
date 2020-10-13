"""
solver to the ode adjoint problem
y'(t,p) = f(y,t,p)
g = g(y(T,p))
dg/dp = ?

following the notation from the
"""


from scipy.integrate import odeint
import numpy as np

class ode_adjoint:
    """
    the # lines hold in general, but are not implemented
    dynamics with parameter p

    dy/dt = f(t,y,p)
    #y(t0) = y0(p)
    y(t0) = y0

    #with desired observable g = g(y(T),p) at end time T
    with desired observable g = g(y(T)) at end time T

    # in general, but not implemented
    #dg/dp (T) = mu(t0)^* dy0/dp + dg/dp(T) + int_t0^T [ mu^* df/dp] dt
    dg/dp (T) = int_t0^T [ mu^* df/dp] dt

    and corresponding adjoint system

    dmu/dt = - (df/dy)* mu
    mu(T) = dg/dy(T)


    f: (y: Rn,t: R) -> Rn
    y0: Rn
    df_dy: (y: Rn, t:R) -> Rn
    df_dp: (y: Rn, t:R) -> Rn
    dg_dy: Rn
    """
    def __init__(self, f, y0, df_dy, df_dp, dg_dy, ts):
        self.f = f
        self.y0 = y0,
        self.df_dy = df_dy
        self.df_dp = df_dp
        self.dg_dy = dg_dy
        self.ts = ts

    def solve(self):
        y = odeint(self.f, self.y0, self.ts)
        return y

    def ode_mu(self):
        def dmu(mu,t):
            return -(self.df_dy(mu,t).T @ mu)

        muT = self.dg_dy

        mu = odeint(dmu, muT, np.flip(self.ts))
        mu = np.flip(mu, axis=0)

        return mu

    def solve_adjoint(self):
        y = self.solve()
        mu = self.ode_mu()

        dg = np.zeros_like(self.dg_dy)

        for i in range(len(self.ts)-1):
            dt = self.ts[i+1] - self.ts[i]
            dg += dt * self.df_dp(y[i], self.ts[i]).T @ mu[i]

        return dg

from numpy.testing import assert_allclose

def test_ode_adjoint(nt = 10, p = 2):
    # dydt =
    y0 = np.array(1)
    f = lambda y,t: np.array(p * y)
    df_dy = lambda y,t: np.array([p])
    df_dp = lambda y,t: np.array([y])
    dg_dy = np.array([1.])
    ts = np.linspace(0, 1, nt)

    adj = ode_adjoint(f, y0, df_dy, df_dp, dg_dy, ts)

    y = adj.solve()
    dg = adj.solve_adjoint()

    assert_allclose(y[-1], np.exp(p))
    assert_allclose(dg[0], np.exp(p))