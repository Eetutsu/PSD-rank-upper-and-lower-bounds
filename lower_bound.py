from numpy.linalg import matrix_rank
import numpy as np



def rank_based_lower_bound(M):
    lower_bound = (1/2)*np.sqrt(1+8*matrix_rank(M))-(1/2)
    return lower_bound
