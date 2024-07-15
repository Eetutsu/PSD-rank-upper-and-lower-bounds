import numpy as np
import test_matrices
import summary

def normalize_mat(M):
    """Normalizes the rows of a matrix"""
    for row in M:
        summa = sum(row)
        for i in range(len(row)):
            row[i] = row[i] / summa
    return M


def generate_q(M):
    """Generates a probability distribution q with random values"""
    q = []
    for i in range(len(M)):
        q.append(np.random.randint(1,21))
    return normalize(q)


def normalize(q_temp):
    """normalizes the entries of a vector to sum up to 1"""
    q = []
    for elem in q_temp:
        q_elem = abs(elem/sum(q_temp))
        q.append(q_elem)
    return q


def F(M_i,M_j):
    """calculates the fideilty beteween the ith and jth row"""
    fid_sum = 0
    for k in range(len(M_i)):
        fid_sum += np.sqrt(M_i[k]*M_j[k])
    return fid_sum


def B5(M, eps=0.001, lr =.001, lr_scaler = 0.75):
    M = np.array(M)
    sums = []
    grad = []
    P_k_log = []
    q_log = [0,0]    

    for i in range((M.shape[1])):
        lr = 0.001
        q = generate_q(M)
        q_log[0] = q    
        for iter in range(1000):
            for k in range((len(q))):
                P_k_log.append(calc_B5(M,q,i))
                q[k] = q[k] + eps
                P_k_log.append(calc_B5(M,q,i))
                grad.append((P_k_log[-2]-P_k_log[-1])/eps)
            for x in range(len(q)):
                q[x] = q[x] + lr*grad[x]
            q = normalize(q)
            q_log[1] = q   
            q_temp = np.array([q_log[0][i] - q_log[1][i] for i in range(len(q))])   
            if(max(q_temp)<0.00001): break     
            lr = lr * lr_scaler
            grad.clear()
        sums.append(max(P_k_log))
    return sum(sums)


def calc_B5(M,q,i):
    summa1 = 0
    for s in range(len(q)):
        for t in range (len(q)):
            summa1 = summa1 + q[s]*q[t]*F(M[s],M[t])**2
    k = 0
    summa2 = 0
    while(k<len(q)):
        summa2 = summa2 + np.dot(q[k],M[k][i])
        k+=1
    return summa2/np.sqrt(summa1)


