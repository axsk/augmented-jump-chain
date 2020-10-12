# OLD softmin, did not really work as supposed
def softmin_p(xs, p=-40):
    "goes to minimum for negative alpha -> -infty "
    return np.sum(xs ** p) ** (1/p)

def dsoftmin_p(xs, p=-40):
    #outer = 1/p * (np.sum(xs ** p)) ** (1/p - 1)
    #inner = p * xs ** (p-1)
    return np.sum(xs ** p) ** (1/p - 1) * (xs ** (p-1))

SOFTMIN_REL_SCALE = 26

# wrapper around the relative softmin
def softmin(xs, scale=SOFTMIN_REL_SCALE):
    return softmin_rel(xs, scale)

def dsoftmin(xs, scale=SOFTMIN_REL_SCALE):
    return fd_softmin_rel(xs, scale)[1]

# scale invariant version of softmin_e due to log/exp transformation
# the scale paramter controls the crispness of the softmin
def softmin_rel(xs, scale=1):
    return np.exp(softmin_e(np.log(xs) * scale) / scale)

def fd_softmin_rel(xs, scale=1):
    ls = np.log(xs) * scale
    sm, dsm = fd_softmin_e(ls)
    sr = np.exp(sm / scale)
    dsr = sr * dsm / xs
    return sr, dsr

# average of xs weighted with softmax(-xs), i.e. the closeness to the minimum
# this function is translation invariant
def softmin_e(xs):
    smax = softmax(-xs)
    return smax @ xs

def fd_softmin_e(xs):
    smax = softmax(-xs)
    smin = smax @ xs
    return smin, smax * (1+smin - xs)

# the classical softmax function
def softmax(xs):
    e = np.exp(xs)
    smax = e / np.sum(e)
    return smax