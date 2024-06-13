from numpy.linalg import matrix_rank
import numpy as np



def rank_based_lower_bound(M):
    """"POSITIVE SEMIDEFINITE RANK" https://arxiv.org/pdf/1407.4095 Proposition 2.5 Page 4"""
    lower_bound = (1/2)*np.sqrt(1+8*matrix_rank(M))-(1/2)
    return lower_bound

def B4(M):
    """"Some upper and lower bounds on PSD-rank" https://arxiv.org/pdf/1407.4308 Theorem 24 Page 10"""
    #Differs from paper such thath instead of the matrix being column stochastic we use row stochastic matrices
    #Check if row stochastic
    epsilon = 0.0001
    rowsum = 0
    for row in M:
        for i in row:
            rowsum += i
        if 1-rowsum>=epsilon or rowsum>1:
            break
        else: rowsum = 0
    if rowsum != 0:
        return "Not a row stochastic matrix"
    #Calculate lower bound
    else:
        sum = 0.0
        maxes = [max(row) for row in M]
        for i in maxes:
            sum += i
        return sum