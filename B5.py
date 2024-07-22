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


def B5D(M, eps=0.001, lr =.001, lr_scaler = 0.95):
    """Calculates a lower bound on the PSD-rank for a row stochastic matrix using gradient descent.

    lower bound calculated using the method found in: https://arxiv.org/pdf/1407.4308 Page 10 Definition 28

    Parameters
    ----------
    M : list
        the matrix we want to find a PSD-rank lower bound for 
    eps : float
        used to approximate the derivate (default 0.001)
    lr : float
        the learning rate used when iterating (default 0.01)
    lr_scaler : float
        scales the learning rate after each iteration (default 0.95)
    
    Returns
    ----------
    float
        the lower bound
    """

    D = np.zeros((len(M),len(M)))
    for i in range(len(D)):
        D[i,i] = np.random.randint(1,11)

    M = np.array(M)
    sums = []
    summed = []
    grad = []
    P_k_log = []
    q_log = [0,0]    

    for iter in range(5):
        for i in range((M.shape[1])):
            lr = 0.001
            q = generate_q(M)
            q_log[0] = q    
            for iter in range(1000):
                for k in range((len(q))):
                    P_k_log.append(calc_B5D(M,q,i,D))
                    q[k] = q[k] + eps
                    D[k,k] = D[k,k] + eps
                    P_k_log.append(calc_B5D(M,q,i,D))
                    grad.append((P_k_log[-1]-P_k_log[-2])/eps)
                for x in range(len(q)):
                    q[x] = q[x] + lr*grad[x]
                    D[x,x] = D[x,x] + lr*grad[x]
                q = normalize(q)
                M = np.dot(D,M)
                M = normalize_mat(M)
                q_log[1] = q   
                q_temp = np.array([q_log[0][i] - q_log[1][i] for i in range(len(q))])   
                if(max(q_temp)<0.00001): break     
                lr = lr * lr_scaler
                grad.clear()
            sums.append(max(P_k_log))
        summed.append(sum(sums))
        sums.clear()
        P_k_log.clear()
    return max(summed)


def calc_B5D(M,q,i,D):
    summa1 = 0
    M = np.dot(D,M)
    for s in range(len(q)):
        for t in range (len(q)):
            summa1 = summa1 + q[s]*q[t]*F(M[s],M[t])**2
    k = 0
    summa2 = 0
    while(k<len(q)):
        summa2 = summa2 + np.dot(q[k],M[k][i])
        k+=1
    return summa2/np.sqrt(summa1)


for matrix in test_matrices.matrices.keys():
    print(f"{B5D(test_matrices.matrices[matrix])} {matrix}")