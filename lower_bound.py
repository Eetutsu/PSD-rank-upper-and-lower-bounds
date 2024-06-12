from numpy.linalg import matrix_rank
import numpy as np


#Lower bound found in https://arxiv.org/pdf/1407.4095
def rank_based_lower_bound(M):
    lower_bound = (1/2)*np.sqrt(1+8*matrix_rank(M))-(1/2)
    return lower_bound

def B4(M):
    epsilon = 0.0001
    rowsum = 0
    for row in M:
        for i in row:
            rowsum += i
        if 1-rowsum>=epsilon:
            break
        else: rowsum = 0
    if rowsum != 0:
        return "Not a row stochastic matrix"
    else:
        sum = 0.0
        maxes = [max(row) for row in M]
        for i in maxes:
            sum += i
        return sum