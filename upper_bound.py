import numpy as np

#Upper bound found in https://arxiv.org/pdf/1407.4095
def mindim_upper_bound(M):
    dims = np.shape(M)
    return min(dims)