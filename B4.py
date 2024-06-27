import random
import numpy as np
import test_matrices


def normalize_mat(M):
    for row in M:
        summa = sum(row)
        for i in range(len(row)):
            row[i] = row[i] / summa
    return M


def B4(M,lr = 0.01, eps = 0.01):   
    sums = []
    grad = []
    p_log = [0,0]
    M = np.array(M).T
    D = np.zeros((M.shape[0],M.shape[0]))
    for iter1 in range(10):
        for i in range(M.shape[0]):
            D[i,i] = np.random.randint(1,11)
        for iter2 in range(1000):
            for i in range(len(D)):
                P = np.dot(D,np.array(M)).T
                p_log[0] = P 
                D[i][i] = D[i][i] + eps
                P = np.dot(D,np.array(M)).T
                p_log[1] = P
                normalize_mat(p_log[0])
                normalize_mat(p_log[1])
                for A in p_log:
                    maxes = [max(row) for row in A.T]
                    sum = 0
                    for k in maxes:
                        sum += k
                    sums.append(sum)
                grad.append((sums[-2]-sums[-1])/eps)
            for i in range(len(D)):
                D[i][i] = D[i][i] - lr*grad[i]
            grad.clear()
    return max(sums)


for matrix in test_matrices.matrices.keys():
    lower_bound = B4(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using B4D")