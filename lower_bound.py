from numpy.linalg import matrix_rank
import numpy as np
import math



def normalize_mat(M):
    """Normalizes the rows of a matrix"""
    for row in M:
        summa = sum(row)
        for i in range(len(row)):
            row[i] = row[i] / summa
    return M


def is_stochastic(M):
    """Checks wheter a matrix is row stochastic or not"""
    rowsum = 0
    for row in M:
        for i in row:
            rowsum += i
        if not math.isclose(1, rowsum, rel_tol=1e-06):
            return False
        else: rowsum = 0
    return True


def B1(M):
    """Lower bound found in: POSITIVE SEMIDEFINITE RANK https://arxiv.org/pdf/1407.4095 Proposition 2.5 Page 4

    Parameters    
    ----------    
    M : list
        the matrix we want to find a lower bound on PSD-rank for
    
    Returns
    ----------
    float
        the lower bound            
    """
    return (1/2)*np.sqrt(1+8*matrix_rank(M))-(1/2)      #Calculate and return the lower bound


def B4(M):
    """Lower bound found in: Some upper and lower bounds on PSD-rank https://arxiv.org/pdf/1407.4308 Theorem 24 Page 10

    Parameters
    ----------
    M : list
      the matrix we want to find a lower bound on PSD-rank for
    
    Returns
    ----------
    string
        if not row stochastic
    float
        the lower bound
    """
    
    if not is_stochastic(M):        #Check if row stochastic
        return "Not a row stochastic matrix"
    else:      #Calculate lower bound
        sum = 0.0
        maxes = [max(row) for row in M]
        for i in maxes:
            sum += i
        return sum


def B4D(M,lr = 0.01, eps = 0.01, break_cond = 0.000001):
    """Lower bound found in: Some upper and lower bounds on PSD-rank: https://arxiv.org/pdf/1407.4308 Definition 25 Page 10
    :param M: the matrix we want to find a lower bound on PSD-rank for
    :type M: 2D array
    :param lr: learning rate used to update the gradient
    :type lr: float
    :param eps: used to approximate the derivate
    :type eps: float
    :param break_cond: used to determine wheter iterating further is senseible
    :type break_cond: float
    :returns: a lower bound for PSD-rank
    :rtype: float
    """
    sums = []
    grad = []
    p_log = [0,0]
    M = np.array(M).T
    #Initialize the nonnegative diagonal matrix D used to scale the original matrix M
    D = np.zeros((M.shape[0],M.shape[0]))
    for iter1 in range(10):
        #Add random nonnegative integers to the diagonal of D
        for i in range(M.shape[0]):
            D[i,i] = np.random.randint(1,11)
        for iter2 in range(1000):
            for i in range(len(D)):
                #Generate the first matrix we use to approximate the gradient vector
                P = np.dot(D,np.array(M)).T
                p_log[0] = P 
                #Add epsilon to the diagonal of D so that we can approximate the derivate
                D[i][i] = D[i][i] + eps
                #Generate second matrix used to approximate the gradient
                P = np.dot(D,np.array(M)).T
                p_log[1] = P
                #normalize both matrices
                normalize_mat(p_log[0])
                normalize_mat(p_log[1])
                #Calculate B4 for both matrices P_0 and P_1
                for A in p_log:
                    maxes = [max(row) for row in A.T]
                    sum = 0
                    for k in maxes:
                        sum += k
                    sums.append(sum)
                #Approximate the derivate and add it to the gradient vector
                grad.append((sums[-2]-sums[-1])/eps)
            #check if we want to iterate further
            if(abs(sums[-1]-sums[-2])<break_cond): break
            #Update the diagonals of the matrix D according to the gradient vector
            for i in range(len(D)):
                D[i][i] = D[i][i] + lr*grad[i]
            lr = lr*0.75
            grad.clear()
    #return the lower bound
    return max(sums)


def grad_vec_min(M,q):
    """calculates the gradient vector for B3 in the point q

    Parameters
    ----------
    M : list
      the matrix used to calculate the fidelity
    q : list
      the vector that determines the point in which the gradient is calculated
    
    Returns
    ----------
    gradient : list
        the gradient vector
    """
    M = np.array(M)
    sum = 0
    gradient = []
    for i in range(len(M)):
        j = i
        while(j in range(len(M))):
            if i == j:
                sum += 2*q[i]+F(M[i],M[j])**2 #The derivate when i = j 
                j+=1
            else:
                sum += 2*q[j]+F(M[i],M[j])**2 #The derivate when i != j
                j+=1
        gradient.append(sum) #the ith entry of the gradient vector
        sum = 0
    return gradient


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
    """calculates the fideilty beteween the ith and jth row as defined in: https://link.springer.com/article/10.1007/s10107-016-1052-0  Page 499
    :param M_i: ith row of the matrix M
    :type M_i: 1D array
    :param M_j: jth row of the matrix M
    :type M_j: 1D array
    :returns: the fidelity between M_i and M_j
    :rtype: float
    """
    
    fid_sum = 0
    for k in range(len(M_i)):
        fid_sum += np.sqrt(M_i[k]*M_j[k])
    return fid_sum



def normalize(q_temp):
    """normalizes the entries of a vector to sum up to 1
    :param q_temp: the vector we want to normalize
    :type q_temp: 1D array
    :returns: a matrix with entries that sum up to one
    :rtype: 1D array
    """
    q = []
    for elem in q_temp:
        q_elem = abs(elem/sum(q_temp))
        q.append(q_elem)
    return q


def generate_q(M):
    """Generates a probability distribution q with random values
    :param M: determines the size of q
    :type M: 2D array
    :reutrns: probability distribution q
    :rtype: 1D array
    """
    q = []
    for i in range(len(M)):
        q.append(np.random.randint(1,21))
    return normalize(q)


def B3_gradient(M, lr=0.001, max_iter = 1000, lr_scaler = 0.75, eps = 0.00001):
    """
    Calculates the lower bound of PSD-rank for the matrix M using B3(P) found in: Some upper and lower bounds on PSD-rank https://arxiv.org/pdf/1407.4308 Page 9 Definition 18
    :param M: The matrix that we find the lower boud for
    :type M: 2-D array
    :param lr: learning rate used in updating the gradient vector
    :type lr = float
    :param: max_iter: maximum iterations for updating the gradient vector (default is 0.001)
    :type max_iter: int
    :param lr_scaler: scales the learning rate after each iteration
    :type lr_scaler: float
    :param eps: stop iterating if change between the entries of the vectors q_i and q_(i+1) is less than eps
    :type eps: float
    :returns: a lower bound of PSD-rank for the matrix M
    :rtype: float
    """
    if not is_stochastic(M): return "Not a row stochastic matrix"
    else:
        #Initialize array that logs results
        res_log = []

        #Initialize array that logs vectors q_i and q_(i+1)
        q_log = [0,0]    
        for i in range(100):
            #Initialize result variable 
            res = 0     
            #Generate a random probability distribution q
            q = generate_q(M)
            #Log q_i   
            q_log[0] = q    
            #Calculate the the gradient vector in the point q_i
            gradient = grad_vec_min(M,q)    
            for ii in range(max_iter):
                for i in range(len(q)):
                    #Update the value of q using the gradient vector
                    q[i] = q[i]-lr*gradient[i]  
                    #Keeps q within bounds
                    if q[i]<0: q[i] = 0 
                #Update the step size (learning rate)    
                lr = lr * lr_scaler 
                #Normalize q so that its entries sum up to one    
                q = normalize(q)  
                #log vector q_(i+1)  
                q_log[1] = q   
                q_temp = np.array([q_log[0][i] - q_log[1][i] for i in range(len(q))])   
                #Check if change is so small that interating further is not sensible
                if(max(q_temp)<eps): break     
                #vector q_i+1 is now q_i
                q_log[0] = q_log[1]
                #Calculate the the gradient vector in the point q_i
                gradient = grad_vec_min(M,q)
            #Calculate the lower bound
            for i in range(len(M)):
                for j in range(len(M)):
                    res = res + q[i]*q[j]*F(M[i],M[j])**2
            res = 1 / res
            #Log the result
            res_log.append(res)
        #Return the lower bound
        return max(res_log)
    

def B3_gradientD(M, lr=0.001, max_iter = 1000, lr_scaler = 0.75, eps = 0.00001):
    """Some upper and lower bounds on PSD-rank https://arxiv.org/pdf/1407.4308 Page 9 Definition 20"""
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
    try:
        ret = np.dot(np.linalg.inv(Hessian),grad_vec_min(M,q))
    except:
        ret = 0
    return ret


def B3_newton(M,lr=0.01,eps = 0.000001):
    q_log = [0,0]
    res = 0
    res_log = []
    for iter in range (10):
        q = generate_q(M)
        q_log[0] = q
        for ii in range(10000):
            newton = (newton_iter(M,q))
            if isinstance(newton, int):
                return 0
            for i in range(len(q)):
                q[i] = q[i] - lr*newton[i]
                lr = lr *0.75
                if q[i]<0: q[i] = 0
            q = normalize(q)
            q_log[1] = q
            q_temp = np.array([q_log[0][i] - q_log[1][i] for i in range(len(q))]) 
            if(max(q_temp)<eps): break
            q_log[0] = q_log[1]
        for i in range(len(M)):
            for j in range(len(M)):
                res = res + q[i]*q[j]*F(M[i],M[j])**2
        res = 1 / res
        res_log.append(res)
    return max(res_log)


l_bounds = [B1, B3_gradient,B3_gradientD,B3_newton, B4, B4D]