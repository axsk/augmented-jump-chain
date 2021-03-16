from jax import jit, jacfwd, jacrev, grad, jvp, vjp, vmap, hessian


import jax.numpy as np

def u(x):
    return x[0] ** 2 + x[1] ** 3

def loss(f, x):
    du = grad(u)(x)
    j = jvp(f, (x,), (du,))[0]
    h = np.trace(hessian(f)(x))
    return j + h


def nn(W):
    def result(x):
        return np.sum(W @ x)
    return result

W = np.array([[1,.2],[3,4]])
x = np.array([5,6.])

def modelloss(W):
    return loss(nn(W), x)

grad(modelloss)(W)