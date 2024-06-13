import numpy as np
from numpy.linalg import matrix_rank


def mindim_upper_bound(M):
    """"POSITIVE SEMIDEFINITE RANK" https://arxiv.org/pdf/1407.4095 Proposition 2.5 Page 4"""
    dims = np.shape(M)
    return min(dims)


def hadamard_sqrt_upper_bound(M):
    """"POSITIVE SEMIDEFINITE RANK" https://arxiv.org/pdf/1407.4095 Corollary 5.3 Page 18"""
    row = 0
    col = 0
    sqrt_ranks = []
    for k in range(1000):
        m = np.array(M)
        for i in m:
            for j in i:
                rng = np.random.randint(0,2)
                if rng == 0:
                    m[row,col] = -np.sqrt(j)
                    col += 1
                else: 
                    m[row,col] = np.sqrt(j)
                    col += 1
            col = 0
            row += 1
        sqrt_ranks.append(matrix_rank(m))
        col = row = 0
    return min(sqrt_ranks)