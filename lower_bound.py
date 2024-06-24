from numpy.linalg import matrix_rank
import numpy as np
from scipy.optimize import minimize

M = ([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])

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
    
    if not is_stochastic(M):        #Check if row stochastic
        return "Not a row stochastic matrix"
    elif not is_D:      #Calculate lower bound
        sum = 0.0
        maxes = [max(row) for row in M]
        for i in maxes:
            sum += i
        return sum
    else:       #Calculate lower bound
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
    

def grad_vec_min(M,q):
    """calculates the gradient vector"""
    M = np.array(M)
    sum = 0
    gradient = []
    for i in range(len(M)):
        j = i
        while(j in range(len(M))):
            if i == j:
                sum += 2*q[i]+F(M[i],M[j])**2
                j+=1
            else:
                sum += 2*q[j]+F(M[i],M[j])**2
                j+=1
        gradient.append(sum)
        sum = 0
    return gradient


def F(M_i,M_j):
    """calculates the fideilty beteween the ith and jth column"""
    fid_sum = 0
    for k in range(len(M_i)):
        fid_sum += np.sqrt(M_i[k]*M_j[k])
    return fid_sum



def normalize(q_temp):
    """normalizes the entries of a vector"""
    q = []
    for elem in q_temp:
        q_elem = abs(elem/sum(q_temp))
        q.append(q_elem)
    return q


def generate_q(M):
    """Generates a probability distribution q with random values"""
    q = []
    for i in range(len(M)):
        q.append(np.random.randint(1,21))
    return normalize(q)


def B3_gradient(M, lr=0.001):
    """"Some upper and lower bounds on PSD-rank" https://arxiv.org/pdf/1407.4308 Page 9 Definition 18"""
    if not is_stochastic(M): return "Not a row stochastic matrix"
    else:
        res_log = []
        for i in range(10): 
            res = 0
            q = generate_q(M)
            gradient = grad_vec_min(M,q)
            for ii in range(100):
                for i in range(len(q)):
                    q[i] = q[i]-lr*gradient[i]
                    if q[i]<0: q[i] = 0
                    #elif q[i]>1: q[i] = 1
                lr = lr * 0.9
                q = normalize(q)
                gradient = grad_vec_min(M,q)
            for i in range(len(M)):
                for j in range(len(M)):
                    res = res + q[i]*q[j]*F(M[i],M[j])**2
            res = 1 / res
            res_log.append(res)
        return max(res_log)