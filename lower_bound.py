from numpy.linalg import matrix_rank
import numpy as np
import random


def normalize_mat(M):
    """Normalizes the rows of a matrix"""
    for row in M:
        summa = sum(row)
        for i in range(len(row)):
            row[i] = row[i] / summa
    return M


def is_stochastic(M):
    """Checks wheter a matrix is row stochastic or not"""
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
    return (1/2)*np.sqrt(1+8*matrix_rank(M))-(1/2)      #Calculate and return the lower bound


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
        arr = [-0.01,0.01]
        sums = [0,0]
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
                    rand = random.choice(arr)
                    D[i][i] = D[i][i] + rand
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
                    if sums[-1]<sums[-2]: continue
                    else: D[i][i] = D[i][i] - 2*rand
        return max(sums)
    

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
    """calculates the fideilty beteween the ith and jth row"""
    fid_sum = 0
    for k in range(len(M_i)):
        fid_sum += np.sqrt(M_i[k]*M_j[k])
    return fid_sum



def normalize(q_temp):
    """normalizes the entries of a vector to sum up to 1"""
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


def B3_gradient(M, lr=0.001, max_iter = 1000, lr_scaler = 0.75, eps = 0.00001):
    """"Some upper and lower bounds on PSD-rank" https://arxiv.org/pdf/1407.4308 Page 9 Definition 18"""
    if not is_stochastic(M): return "Not a row stochastic matrix"
    else:
        res_log = []    #Log containing results
        q_log = [0,0]   #Log containing q_i and q_(i+1) 
        for i in range(100): 
            res = 0     #Initialize result variable
            q = generate_q(M)   #Generate a random probability distribution
            q_log[0] = q    #Log q_i
            gradient = grad_vec_min(M,q)    #Calculate the the gradient vector in the point q
            for ii in range(max_iter):
                for i in range(len(q)):
                    q[i] = q[i]-lr*gradient[i]  #Update the value of q according to the gradient
                    if q[i]<0: q[i] = 0     #Keeps q in bounds
                lr = lr * lr_scaler     #Update the step size (learning rate)
                q = normalize(q)    #Normalize q so that its entries sum up to one
                q_log[1] = q    #log q_(i+1)
                q_temp = np.array([q_log[0][i] - q_log[1][i] for i in range(len(q))])   
                if(max(q_temp)<eps): break      #Check if change is so small that interating further is not sensible
                q_log[0] = q_log[1]
                gradient = grad_vec_min(M,q)
            for i in range(len(M)):
                for j in range(len(M)):
                    res = res + q[i]*q[j]*F(M[i],M[j])**2
            res = 1 / res
            res_log.append(res)
        return max(res_log)