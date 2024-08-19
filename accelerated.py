import numpy as np
from summary import solve
import test_matrices


def generate_A_B(M, dim):
    """
    Intialize PSD factors for M

    Parameters
    -----------
    M : list
        Determines the size of sets A and B
    dim
        matrices A and B are of size dim x dim

    
    Returns
    ----------
    list
        sets of matrices A and B
    """
    arr_A = [] #Array of matrices A
    arr_B = [] #Array of matrices B 
    for i in range(M.shape[0]): #Generate A matrices
        mat = np.random.rand(dim, dim)
        arr_A.append(np.dot(mat, mat.T))    #Ensures the matrcies are symmetric
    for j in range(M.shape[1]): #Generate B matrices
        mat = np.random.rand(dim, dim)
        arr_B.append(np.dot(mat, mat.T))    #Ensures the matrcies are symmetric
    return arr_A, arr_B



def accelerated_gradinet(M):
    delta = 5
    M = np.array(M)
    dim = solve(M, print_steps=2)   #All the possible PSD-ranks
    for iter in range(len(dim)):    #Iterate for every possilbe PSD-rank
        arr_A, arr_B = generate_A_B(M, dim[iter])   #Intialize A and B matrices
        A = flatten(arr_A)
        B = flatten(arr_B)
        scaling = sum(sum((M@B.T)*A.T))/sum(sum((B@B.T)*(A@A.T)))
        A = A*scaling
        for iter in range(100):
            AX = B @ M.T; AAt = B@B.T
            A = faststepgrad(A,AX,AAt, delta)
            AX = A@M; AAt = A@A.T
            B = faststepgrad(B,AX,AAt,delta)
    return A,B


def faststepgrad(B,AX,AAt, delta):
    k = int(np.sqrt(AX.shape[0]))
    n = B.shape[1]
    L = max(np.linalg.eigvalsh(AAt))
    Y = B
    for i in range(1, max(1,delta*k)):
        V = B + ((i-2)/(i+1))*(B-Y)
        B = V-gradient(V,AAt,AX)/L
        for j in range(0,n):
            B[:,j] = projection(B[:,j],k)
    return B


def projection(X,k):
    X = np.reshape(X, (k, k))
    V,D =np.linalg.eig(X)
    eigvals_positive = np.diag(np.maximum(V, 0))

    X = D@eigvals_positive@D.T
    return X.flatten()


def gradient(B,AAt,AX):
    return AAt@B-AX



def flatten(arr):
    flattened_columns = [elem.flatten() for elem in arr]
    new_matrix = np.column_stack(flattened_columns)
    return new_matrix


print(accelerated_gradinet(test_matrices.A))