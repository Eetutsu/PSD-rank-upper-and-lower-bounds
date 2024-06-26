import random
import numpy as np



def normalize_mat(M):
    for row in M:
        summa = sum(row)
        for i in range(len(row)):
            row[i] = row[i] / summa
    return M


def B4(M):   
    arr = [-0.1,0.1]
    sums = [0,0]
    p_log = [0,0]
    M = np.array(M).T
    D = np.zeros((M.shape[0],M.shape[0]))
    for i in range(M.shape[0]):
        D[i,i] = np.random.randint(1,11)
    P = np.dot(D,np.array(M)).T
    for iter in range(50):
        for i in range(len(D)):
            p_log[0] = P 
            rand = random.choice(arr)
            D[i][i] = D[i][i] + rand
            P = np.dot(D,np.array(M)).T
            p_log[1] = P
            normalize_mat(p_log[0])
            normalize_mat(p_log[1])
            j = 0
            for A in p_log:
                maxes = [max(row) for row in A.T]
                sum = 0
                for k in maxes:
                    sum += k
                sums[j] = sum
                j+=1
            if sums[0]<sums[1]: continue
            else: D[i][i] = D[i][i] - 2*rand
    return max(sums)


K_plus = ([[1/2,1/2,0,0],[1/2,0,1/2,0],[1/2,0,0,1/2],[0,1/2,0,1/2],[0,0,1/2,1/2]])
print(B4(K_plus))