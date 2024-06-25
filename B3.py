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


def B3_gradient(M, lr=0.01):
    res = 0
    q = generate_q(M)
    gradient = grad_vec_min(M,q)
    for ii in range(100):
        for i in range(len(q)):
            q[i] = q[i]-lr*gradient[i]
            if q[i]<0: q[i] = 0
            elif q[i]>1: q[i] = 1
        lr = lr * 0.75
        q = normalize(q)
        gradient = grad_vec_min(M,q)
    for i in range(len(M)):
        for j in range(len(M)):
            res = res + q[i]*q[j]*F(M[i],M[j])**2
    res = 1 / res
    return res


def newton_iter(M,q):
    arr = []
    for i in range(len(M)):
        arr.append(F(M[i],M[i]))
    return np.dot(np.linalg.inv(arr),grad_vec_min(M,q))


def B3_newtonian(M):
    q = generate_q(M)
    q = q - newton_iter(M,q)

B3_newtonian(K_plus)