from numpy.linalg import matrix_rank
import numpy as np
from scipy.optimize import minimize




def is_stochastic(M):
    epsilon = 0.0001
    rowsum = 0
    for row in M:
        for i in row:
            rowsum += i
        if 1-rowsum>=epsilon or rowsum>1:
            return False
        else: rowsum = 0
    return True

def rank_based_lower_bound(M):
    """"POSITIVE SEMIDEFINITE RANK" https://arxiv.org/pdf/1407.4095 Proposition 2.5 Page 4"""
    lower_bound = (1/2)*np.sqrt(1+8*matrix_rank(M))-(1/2)
    return lower_bound

def B4(M, is_D = False):
    """"Some upper and lower bounds on PSD-rank" https://arxiv.org/pdf/1407.4308 Theorem 24 Page 10"""
    #Differs from paper such that instead of the matrix being column stochastic we use row stochastic matrices
    #Check if row stochastic
    
    if not is_stochastic(M):
        return "Not a row stochastic matrix"
    #Calculate lower bound
    elif not is_D:
        sum = 0.0
        maxes = [max(row) for row in M]
        for i in maxes:
            sum += i
        return sum
    else:
        M = np.array(M)
        D = np.zeros((M.shape[0],M.shape[0]))
        for i in range(M.shape[0]):
            D[i,i] = np.random.randint(1,11)
        P = np.dot(D,np.array(M))
        sum = 0.0
        for row in P:
            index = 0
            rowsum = 0
            for j in row:
                rowsum += j
            for i in row:
                row[index] = i/rowsum
                index += 1
        maxes = [max(row) for row in P]
        for i in maxes:
            sum += i
        return sum


def B3_func(M):
    
    M = np.array(M)
    cols = M.shape[1]
    ii = 0
    kk = 0
    jj = 0
    fidelities = []
    sum = 0
    results = 0
    while(ii<M.shape[0]):
        while(jj<M.shape[0]):
            while(kk<cols):
                sum += np.sqrt(M[ii,kk]*M[jj,kk])
                kk += 1
            fidelities.append(sum**2)
            sum = 0
            kk = 0
            jj += 1
        jj = 0
        ii += 1
    return fidelities

    

def B3(M): 
    if not is_stochastic(M):
        return "Not a row stochastic matrix"
    else:
        return 0.5*sum(B3_func(M))