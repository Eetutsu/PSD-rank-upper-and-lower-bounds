import numpy as np
import test_matrices
import picos as pic
from summary import solve

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


def alternating_strategy(M,round_accuracy=10):
    """
    Heuristic method for approximating a PSD-factorization for the matrix M

    Algorithm found in: Algorithms for Positive Semidefinite Factorization https://arxiv.org/pdf/1707.07953 Page 4 Algorithm 1 

    Parameters
    ----------
    M : list
        the matrix we want to find a PSD-factorization for
    round_accuracy : int
        amount of decimals when rounding the values of the matrices (default=10)
    """


    M = np.array(M)
    dim = solve(M, print_steps=2)   #All the possible PSD-ranks
    for iter in range(len(dim)):    #Iterate for every possilbe PSD-rank    
        arr_A, arr_B = generate_A_B(M, dim[iter])   #Intialize A and B matrices
        for i in range(100):
            try:
                arr_A = optimize_subproblem(arr_A,M,arr_B)  #Optimize A matrices using B matrices
                arr_B = optimize_subproblem(arr_B,M.T,arr_A)    #Optimize B matrice using A matrices
            except:
                print("Division by zero")   #Picos sometimes divides by zero    
                break
        X = np.zeros((M.shape[0],M.shape[1]))   #Generate matrix from factors
        for i in range(len(arr_A)):
            for j in range(len(arr_B)):
                X[i ,j] = np.trace(np.dot(arr_A[i],arr_B[j]))
        print("A eigenvalues and matrices")     #Print factros from A and their eigenvalues
        for mat in arr_A:
            print(np.round(mat,round_accuracy))
            print(np.linalg.eigh(mat)[0])
            print("\n")
        print("\n B eigenvalues and matrices")  #Print factros from B and their eigenvalues
        for mat in arr_B:
            print(np.round(mat,round_accuracy))
            print(np.linalg.eigh(mat)[0])    
            print("\n")

        print("Original matrix M:")
        print(np.round(M,round_accuracy))
        print("New matrix X formed from factors: X_ij = Tr(A_iB_j)")
        print(np.round(X,round_accuracy))
        print(f"Frobenius norm (M-X): {np.linalg.norm(M-X)} \n")    #Norm determines how good of a factorization was found


def objective_function(X_row, B, A):
    """
    The function we use to optimize the matix A

    function found in: Algorithms for Positive Semidefinite Factorization https://arxiv.org/pdf/1707.07953 Page 4 Algorithm (5)


    Parameters
    -----------
    X_row : list
        ith row of the original matrix we want to factorize
    B : list
        the matrix we use in optimizing
    A : list
        the matrix we want to optimize

    
    Returns
    ---------
    Quadratic Expression
        objective function to minimize
    """


    return sum((X_row[j] - (A | B[j]))**2 for j in range(len(B)))


def optimize_subproblem(optimized, M, optimizer):
    """
    Optimize the factors {A_1 ... A_m} and {B_1 ... B_n}

    Algorithm found in: Algorithms for Positive Semidefinite Factorization https://arxiv.org/pdf/1707.07953 Page 4 Algorithm 2

    Parameters
    ----------
    optimized : list
        the set of matrices we want to optimize
    M : list
        the original matrix we use in optimizing
    optimizer : list
        the set of matrices we use to optimize
    
    
    Retruns
    -------
    list
        the set of optimized arrays
    """


    n = optimized[0].shape[0]  # Assuming all matrices in optimized are of the same shape
    A  = pic.HermitianVariable('A', (n, n))     #PSD-matrix, so we want the factors to be hermitian
    problem = pic.Problem()         # Create a Picos problem
    problem.add_constraint(A >> 0)  #Eigenvalues nonnegative
    for i in range(len(optimized)):
        X_row = M[i, :] #ith row of the original matrix we want to factorize
        A.value = optimized[i]
        objective = objective_function(X_row, optimizer, A) #The function we use to optimize
        problem.set_objective('min', objective) #minimize the objective function§
        problem.solve() 
        optimized[i] = np.array(A.value)    #replace old value with optimized value
        
    return optimized
        


    
                
                    
def accelerated_gradient_method(M):


    def optimize_subproblem(M,A,B, dim):
        A_log = [A]
        for t in range(1,len(A)):
            Y = A_log[t-1]+((t-2)/(t+1))*(A_log[t-1]-A_log[t-2])
            A = Proj_Q(Y, B, gradient(M,B,Y) ,M,dim)
            A_log.append(A)
        return A_log[-1]
            


    def gradient(X,B,Y):
        return np.dot(X,B.T)-np.linalg.multi_dot([Y.T,B,B.T])
        

    def Proj_Q(Y,B, gradient,M, dim):
        L = max(np.linalg.eigvalsh(np.dot(B,B.T)))
        C = Y + (1/L)*gradient.T

        arr_C = unflatten(C, dim)
        temp = []
        for C in arr_C:
            eigvals, eigvecs = np.linalg.eigh(C)
            
            eigvals_positive = np.diag(np.maximum(eigvals, 0))
            
            projected_C = np.linalg.multi_dot([eigvecs,eigvals_positive, eigvecs.T])
            temp.append(projected_C)
        
        return flatten(temp)


    def flatten(arr):
        flattened_columns = [elem.flatten() for elem in arr]
        new_matrix = np.column_stack(flattened_columns)
        return new_matrix


    def unflatten(new_matrix, dim):
        num_columns = new_matrix.shape[1]
        
        original_matrices = [new_matrix[:, i].reshape((dim,dim)) for i in range(num_columns)]
        
        return original_matrices



    M = np.array(M)
    dim = solve(M, print_steps=2)
    for iter in range(len(dim)):
        arr_A, arr_B = generate_A_B(M, dim[iter])
        A = flatten(arr_A)
        B = flatten(arr_B)
        for i in range(1000):
            A = optimize_subproblem(M,A,B, dim[iter])
            B = optimize_subproblem(M,B,A, dim[iter])
        arr_A = unflatten(A,dim[iter])
        arr_B = unflatten(B,dim[iter])
        X = np.zeros((M.shape[0],M.shape[1]))
        for i in range(len(arr_A)):
            for j in range(len(arr_B)):
                X[i ,j] = np.trace(np.dot(arr_A[i],arr_B[j]))
        print("A eigenvalues and matrices")
        for mat in arr_A:
            print(mat)
            print(np.linalg.eigh(mat)[0])
            print("\n")
        print("\n B eigenvalues and matrices")
        for mat in arr_B:
            print(mat)
            print(np.linalg.eigh(mat)[0])    
            print("\n")
        print("Original matrix and new matrix:")
        print(M)
        print(X)
        print(f"Frobenius norm: {np.linalg.norm(M-X)} \n")
        




#alternating_strategy([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
#alternating_strategy([[0.47619048, 0.23809524, 0, 0.28571429,],[0.875, 0,    0,   0.125],[0.3, 0.26666667,0.16666667,0.26666667],[0.88888889, 0, 0, 0.11111111],[0.38095238, 0.14285714, 0, 0.47619048]])
#for matrix in test_matrices.matrices.values():
#    alternating_strategy(matrix)
#    print("\n")
accelerated_gradient_method(test_matrices.A)