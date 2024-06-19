import numpy as np

K_plus = ([[1/2,1/2,0,0],[1/2,0,1/2,0],[1/2,0,0,1/2],[0,1/2,0,1/2],[0,0,1/2,1/2]])






def grad_vec_min(M,q):
    """calculates the gradiant vector"""
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
        q_elem = elem/sum(q_temp)
        q.append(q_elem)
    return q


def generate_q(M):
    """Generates a probability distribution q"""
    q = []
    for i in range(len(M)):
        q.append(np.random.randint(1,21))
    return normalize(q)


def B3(M, lr=0.01):
    q = generate_q(M)
    gradient = grad_vec_min(M,q)
    for i in range(1000):
        for i in range(len(q)):
            q[i] = q[i]-lr*gradient[i]
        normalize(q)
        gradient = grad_vec_min(M,q)
    return gradient


B3(K_plus)