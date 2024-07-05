import numpy as np
import test_matrices

K_plus = ([[1/2,1/2,0,0],[1/2,0,1/2,0],[1/2,0,0,1/2],[0,1/2,0,1/2],[0,0,1/2,1/2]])
NP = ([[1,0,0,25],[0,1,0,144],[0,0,1,169],[1,1,1,0]])
A_2 = ([[0,0,1,2,2,1],[1,0,0,1,2,2],[2,1,0,0,1,2],[2,2,1,0,0,1],[1,2,2,1,0,0],[0,1,2,2,1,0]])





def grad_vec_minD(M,q):
    """calculates the gradient vector"""
    M = np.array(M)
    sum = 0
    D = np.zeros((M.shape[1],M.shape[1]))
    for i in range(M.shape[1]):
        D[i,i] = np.random.randint(1,11)
    gradient = []
    for i in range(len(q)):
        for j in range(len(q)):
            for k in range(len(D)):
                l = k
                while(l in range(len(D))):
                    if l == k:
                        sum = sum + q[i]*q[j]*2*D[k][k]*M[i][k]*M[j][k]
                        l += 1
                    else:
                        sum = sum + q[i]*q[j]*2*D[l][l]*np.sqrt(M[i][k]*M[j][k])*np.sqrt(M[i][l]*M[j][l])
                        l+=1
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


    

def B3_gradientD(M, lr=0.001, max_iter = 1000, lr_scaler = 0.75, eps = 0.00001):
    """"Some upper and lower bounds on PSD-rank" https://arxiv.org/pdf/1407.4308 Page 9 Definition 18"""
    if not is_stochastic(M): return "Not a row stochastic matrix"
    else:
        res_log = []    #Log containing results
        q_log = [0,0]   #Log containing q_i and q_(i+1) 
        for i in range(100): 
            res = 0     #Initialize result variable
            q = generate_q(M)   #Generate a random probability distribution
            q_log[0] = q    #Log q_i
            gradient = grad_vec_minD(M,q)    #Calculate the the gradient vector in the point q
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
                gradient = grad_vec_minD(M,q)
            for i in range(len(M)):
                for j in range(len(M)):
                    res = res + q[i]*q[j]*F(M[i],M[j])**2
            res = 1 / res
            res_log.append(res)
        return max(res_log)


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

for matrix in test_matrices.matrices.keys():
    lower_bound = B3_gradientD(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using gradient method")
