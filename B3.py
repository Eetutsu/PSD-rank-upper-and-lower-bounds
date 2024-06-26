import numpy as np

K_plus = ([[1/2,1/2,0,0],[1/2,0,1/2,0],[1/2,0,0,1/2],[0,1/2,0,1/2],[0,0,1/2,1/2]])
NP = ([[1,0,0,25],[0,1,0,144],[0,0,1,169],[1,1,1,0]])
A_2 = ([[0,0,1,2,2,1],[1,0,0,1,2,2],[2,1,0,0,1,2],[2,2,1,0,0,1],[1,2,2,1,0,0],[0,1,2,2,1,0]])





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



def newton_iter(M,q):
    Hessian = np.zeros((len(q),len(q)))
    for i in range(len(Hessian)):
        j = 0
        while(j<len(Hessian)):
            if i == j:
                Hessian[i][j] = 2*F(M[i],M[j])**2
                j += 1
            else:
                Hessian[i][j] = 2*F(M[i],M[j])**2
                j+=1
    return np.dot(np.linalg.inv(Hessian),grad_vec_min(M,q))


def B3_newton(M,lr=0.01):
    res = 0
    res_log = []
    for iter in range (10):
        q = generate_q(M)
        for ii in range(10000):
            newton = (newton_iter(M,q))
            for i in range(len(q)):
                q[i] = q[i] - lr*newton[i]
                lr = lr *0.75
                if q[i]<0: q[i] = 0
            normalize(q)
        for i in range(len(M)):
            for j in range(len(M)):
                res = res + q[i]*q[j]*F(M[i],M[j])**2
        res = 1 / res
        res_log.append(res)
    return max(res_log)
    


B3_newton(K_plus)