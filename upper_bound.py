import numpy as np
from numpy.linalg import matrix_rank
from itertools import product



def mindim_upper_bound(M):
    """"POSITIVE SEMIDEFINITE RANK" https://arxiv.org/pdf/1407.4095 Proposition 2.5 Page 4"""
    dims = np.shape(M)
    return min(dims)


def hadamard_sqrt_upper_bound(M, is_accurate = True):
    """"POSITIVE SEMIDEFINITE RANK" https://arxiv.org/pdf/1407.4095 Corollary 5.3 Page 18"""
    M = np.array(M)

    row = 0
    col = 0
    sqrt_ranks = []
    if len(M)*len(M.T)>=20:
        is_accurate = False
        print("Matrix too big for all combinations, using stochastic method")

    if not is_accurate:     #Stochastic method
        for k in range(100000):
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
    else:       #Every combination
        p, q = M.shape
        possible_values = [np.array([np.sqrt(M[i, j]), -np.sqrt(M[i, j])]) for i in range(p) for j in range(q)]
        all_combinations = product(*possible_values)
        hadamard_square_roots = []
        for combination in all_combinations:
            matrix = np.array(combination).reshape(p, q)
            hadamard_square_roots.append(matrix)
        sqrt_ranks = [np.linalg.matrix_rank(matrix) for matrix in hadamard_square_roots]
    
    return min(sqrt_ranks)


u_bounds = [mindim_upper_bound,hadamard_sqrt_upper_bound]