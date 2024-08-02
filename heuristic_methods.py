import numpy as np
import test_matrices
from scipy.optimize import minimize
from summary import solve

def alternating_strategy(M):
    M = np.array(M)
    dim = solve(M, print_steps=2)
    for iter in range(len(dim)):
        arr_A, arr_B = generate_A_B(M, dim[iter])
        for i in range(10):
            arr_A = optimize_subproblem(arr_A,M,arr_B)
            arr_B = optimize_subproblem(arr_B,M.T,arr_A)
        X = np.zeros((M.shape[0],M.shape[1]))
        for i in range(len(arr_A)):
            for j in range(len(arr_B)):
                X[i ,j] = np.trace(np.dot(arr_A[i],arr_B[j]))
        print("A eigenvalues")
        for mat in arr_A:
            print(np.linalg.eigh(mat)[0])
        print("\n B eigenvalues")
        for mat in arr_B:
            print(np.linalg.eigh(mat)[0])    
        print("\n", M)
        print(X.round(decimals=4))
        


def objective_function(A_flat, X_row, B):
    A = A_flat.reshape((B[0].shape[0], B[0].shape[1]))  # Reshape A to original dimensions
    return np.sum([(X_row[j] - np.trace(np.dot(A, B[j])))**2 for j in range(len(B))])


def optimize_subproblem(optimized, M, optimizer):
    for i in range(len(optimized)):
        X_row = M[i, :]
        A_initial = optimized[i].flatten()
        
        result = minimize(objective_function, A_initial, args=(X_row, optimizer), method='L-BFGS-B')
        
        optimized[i] = result.x.reshape(optimized[i].shape)
        
    return optimized
        
                    
            
def generate_A_B(M, dim):
    arr_A = []
    arr_B = []
    for i in range(M.shape[0]):
        mat = np.random.rand(dim, dim)
        arr_A.append(np.dot(mat.T, mat))
    for j in range(M.shape[1]):
        mat = np.random.rand(dim, dim)
        arr_B.append(np.dot(mat, mat.T))
    return arr_A, arr_B



alternating_strategy([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
#alternating_strategy([[0.47619048, 0.23809524, 0, 0.28571429,],[0.875, 0,    0,   0.125],[0.3, 0.26666667,0.16666667,0.26666667],[0.88888889, 0, 0, 0.11111111],[0.38095238, 0.14285714, 0, 0.47619048]])
#for matrix in test_matrices.matrices.values():
#    alternating_strategy(matrix)
#    print("\n")