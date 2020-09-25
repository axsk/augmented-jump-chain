import numpy as np
import scipy.sparse as sp

def simpleq(n):
    """ rate matrix of uniform 1d diffusion """
    Q = np.zeros((n,n))
    np.fill_diagonal(Q[1:,:], 1)
    np.fill_diagonal(Q[:,1:], 1)
    np.fill_diagonal(Q, -np.sum(Q, axis=1))
    return Q

def adjacency2d(nx, ny):
    """ sparse adjacency matrix for a 2d grid """
    Q = sp.csr_matrix((nx*ny, nx*ny))

    def cartesian_to_linear(i,j):
        return np.ravel_multi_index((j,i), (ny,nx)) 

    for i in range(nx):
        for j in range(ny):
            fr = cartesian_to_linear(i,j)
            if i+1<nx:
                to = cartesian_to_linear(i+1,j)
                Q[fr,to] = 1
            if i>0:
                to = cartesian_to_linear(i-1,j)
                Q[fr,to] = 1
            if j+1<ny:
                to = cartesian_to_linear(i,j+1)
                Q[fr,to] = 1
            if j>0:
                to = cartesian_to_linear(i,j-1)
                Q[fr,to] = 1

    #rowsum = np.array(Q.sum(axis=1)).flatten()
    return Q# - sp.diags(rowsum)

import matplotlib.pyplot as plt
from copy import copy

class sqra2d:
    """ square root approximation on a regular 2d grid """
    def __init__(self, potential, beta=1, phi=1):
        self.ny, self.nx = np.shape(potential)
        self.beta=beta
        self.phi=phi
        self.u = potential
        self.A = adjacency2d(self.nx, self.ny)
        self.Q = sqra(self.u.flatten(), self.A, self.beta, self.phi)
        self.N = self.Q.shape[0]
    
    def perturbed_Q(self, v):
        """ compute the resulting sqra for a perturbation of the original potential"""
        return sqra(self.u.flatten() + v.flatten(), self.A, self.beta, self.phi)

    def perturbed_copy(self, v):
        s = copy(self)
        s.u = self.u + np.reshape(v, self.u.shape)
        s.Q = self.perturbed_Q(v)
        return s

    def plot(self):
        plt.title("SQRA potential")
        plt.imshow(self.u)
        plt.figure()
        plt.title("SQRA generator")
        plt.imshow(self.Q.toarray())
        plt.colorbar()

    
def doublewell2d(nx = 5, ny = 3, xlims=(-1.5,1.5), ylims=(-1.5,1.5)):
    """ evaluation of the 2d double well potential """
    xs = np.linspace(*xlims, nx)
    ys = np.linspace(*ylims, ny) 
    xs, ys = np.meshgrid(xs, ys, sparse=True)

    return (xs**2-1)**2 + ys**2

def test_doublewell2d():
    assert doublewell2d().shape == (3,5)

def sqra(u, A, beta, phi):
    """ Square-root approximation of the generator
    (of the Overdamped Langevin model)

    u: vector of pointwise evaluation of the potential
    A: adjacency matrix of the discretization
    beta: inverse temperature
    phi: the flux constant, determined by the temperature and the discr.
    """

    pi  = np.sqrt(np.exp(- beta * u))  # Boltzmann distribution
    pi /= np.sum(pi)

    D  = sp.diags(pi)
    D1 = sp.diags(1 / pi)
    Q  = phi * D1.dot(A).dot(D)
    rowsum = np.array(Q.sum(axis=1)).flatten() 
    for i in range(len(pi)):
        Q[i,i] = -rowsum[i]
    return Q
