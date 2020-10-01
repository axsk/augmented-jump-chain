import numpy as np
import scipy.sparse as sp

def simpleq(n):
    """ rate matrix of uniform 1d diffusion """
    Q = np.zeros((n,n))
    np.fill_diagonal(Q[1:,:], 1)
    np.fill_diagonal(Q[:,1:], 1)
    np.fill_diagonal(Q, -np.sum(Q, axis=1))
    return Q

def adjacency_ndbox(dims):
    return adjacency_ndtorus(dims, torus = False)

def adjacency_ndtorus(dims, torus = True):
    nd = len(dims)
    N = np.prod(dims)

    neighbours = np.vstack([np.diag(np.ones(nd, dtype=int)), -np.diag(np.ones(nd, dtype=int))])
    singletondim = np.array(dims) == 1
    neighbours[:, singletondim] = 0

    row = np.zeros(N*2*nd, dtype=int)
    col = np.zeros(N*2*nd, dtype=int)
    data = np.ones(N*2*nd, dtype=bool)
    cut = np.zeros(N*2*nd, dtype=bool)

    for i in range(N):
        multi = np.unravel_index(i, dims) # find multiindex of current cell
        mn = multi + neighbours # add neighbour offset
        if not torus:
            cut[i*2*nd:(i+1)*2*nd] = np.sum((mn < 0) + ( mn >= dims), axis=1) # check if boundary is hit
        mn = np.mod(multi + neighbours, dims) # glue together boundary
        neighbour_flat = np.ravel_multi_index(mn.T, dims) # back to flat inds
        #print(neighbour_flat)
        row[i*2*nd:(i+1)*2*nd] = i
        col[i*2*nd:(i+1)*2*nd] = neighbour_flat
 
    if not torus: # cut out the points which were glued at boundaries
        sel = ~cut
        data = data[sel]
        row = row[sel]
        col = col[sel]

    #return row, col
    A = sp.coo_matrix((data, (row, col))).tocsr()
    A[np.diag_indices_from(A)] = False
    return A


def adjacency2d(nx, ny):
    print("adjacency2d is deprecated, use adjacency_nbox")
    return adjacency_ndbox((ny,nx))

class Sqra:
    def __init__(self, u, beta=1, phi=1, torus=False):
        self.u = u
        self.beta = beta
        self.phi = phi

        self.dims = np.shape(u)
        self.A = adjacency_ndtorus(self.dims, torus)
        self.Q = self.sqra_Q()
        self.N = self.Q.shape[0]
    
    def sqra_Q(self):
        return sqra(self.u.flatten(), self.A, self.beta, self.phi)
    
    def perturbed(self, v):
        s = copy(self)
        s.u = self.u + np.reshape(v, self.u.shape)
        s.Q = s.sqra_Q()
        return s

    def plot(self, potential=True, generator=True):
        if potential:
            plt.title("SQRA potential")
            plt.imshow(self.u)
            plt.colorbar()
        if generator:
            plt.figure()
            plt.title("SQRA generator")
            plt.imshow(self.Q.toarray())
            plt.colorbar()

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
        s.Q = sqra(s.u.flatten(), s.A, s.beta, s.phi)
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
