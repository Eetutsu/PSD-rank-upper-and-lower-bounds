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
    arr_A = []  # Array of matrices A
    arr_B = []  # Array of matrices B
    for i in range(M.shape[0]):  # Generate A matrices
        mat = np.random.rand(dim, dim)
        arr_A.append(np.dot(mat, mat.T))  # Ensures the matrcies are symmetric
    for j in range(M.shape[1]):  # Generate B matrices
        mat = np.random.rand(dim, dim)
        arr_B.append(np.dot(mat, mat.T))  # Ensures the matrcies are symmetric
    return arr_A, arr_B


def alternating_strategy(M, round_accuracy=10):
    """
    Heuristic method for approximating a PSD-factorization for the matrix M

    Algorithm found in: Algorithms for Positive Semidefinite Factorization
    https://arxiv.org/pdf/1707.07953 Page 4 Algorithm 1

    Parameters
    ----------
    M : list
        the matrix we want to find a PSD-factorization for
    round_accuracy : int
        amount of decimals when rounding the values of the matrices (default=10)
    """

    M = np.array(M)
    dim = solve(M, print_steps=2)  # All the possible PSD-ranks
    for iter in range(len(dim)):  # Iterate for every possilbe PSD-rank
        arr_A, arr_B = generate_A_B(M, dim[iter])  # Intialize A and B matrices
        for i in range(100):
            try:
                arr_A = optimize_subproblem(
                    arr_A, M, arr_B
                )  # Optimize A matrices using B matrices
                arr_B = optimize_subproblem(
                    arr_B, M.T, arr_A
                )  # Optimize B matrice using A matrices
            except ZeroDivisionError:
                print("Division by zero")  # Picos sometimes divides by zero
                break
        X = np.zeros((M.shape[0], M.shape[1]))  # Generate matrix from factors
        for i in range(len(arr_A)):
            for j in range(len(arr_B)):
                X[i, j] = np.trace(np.dot(arr_A[i], arr_B[j]))
        print(
            "A eigenvalues and matrices"
        )  # Print factros from A and their eigenvalues
        for mat in arr_A:
            print(np.round(mat, round_accuracy))
            print(np.linalg.eigh(mat)[0])
            print("\n")
        print(
            "\n B eigenvalues and matrices"
        )  # Print factros from B and their eigenvalues
        for mat in arr_B:
            print(np.round(mat, round_accuracy))
            print(np.linalg.eigh(mat)[0])
            print("\n")

        print("Original matrix M:")
        print(np.round(M, round_accuracy))
        print("New matrix X formed from factors: X_ij = Tr(A_iB_j)")
        print(np.round(X, round_accuracy))
        print(
            f"Frobenius norm (M-X): {np.linalg.norm(M-X)} \n"
        )  # Norm determines how good of a factorization was found
    return arr_A, arr_B


def objective_function(X_row, B, A):
    """
    The function we use to optimize the matix A

    function found in: Algorithms for Positive Semidefinite Factorization
    https://arxiv.org/pdf/1707.07953 Page 4 Algorithm (5)


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

    return sum((X_row[j] - (A | B[j])) ** 2 for j in range(len(B)))


def optimize_subproblem(optimized, M, optimizer):
    """
    Optimize the factors {A_1 ... A_m} and {B_1 ... B_n}

    Algorithm found in: Algorithms for Positive Semidefinite Factorization
    https://arxiv.org/pdf/1707.07953 Page 4 Algorithm 2

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

    n = optimized[0].shape[
        0
    ]  # Assuming all matrices in optimized are of the same shape
    A = pic.HermitianVariable(
        "A", (n, n)
    )  # PSD-matrix, so we want the factors to be hermitian
    problem = pic.Problem()  # Create a Picos problem
    problem.add_constraint(A >> 0)  # Eigenvalues nonnegative
    for i in range(len(optimized)):
        X_row = M[i, :]  # ith row of the original matrix we want to factorize
        A.value = optimized[i]
        objective = objective_function(
            X_row, optimizer, A
        )  # The function we use to optimize
        problem.set_objective("min", objective)  # minimize the objective function§
        problem.solve()
        optimized[i] = np.array(A.value)  # replace old value with optimized value

    return optimized


def generate_A_B_gradient(M, dim):
    """
    Intialize hermitian PSD factors for M

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

    arr_A = []  # Array of matrices A
    arr_B = []  # Array of matrices B
    for i in range(M.shape[0]):  # Generate A matrices
        real_part = np.random.random((dim, dim))  # Random values between 0 and 1
        imag_part = np.random.random((dim, dim))  # Random values between 0 and 1
        mat = real_part + 1j * imag_part
        mat = mat + mat.conjugate().T  # Ensures the matrcies are symmetric
        arr_A.append(mat)
    for j in range(M.shape[1]):  # Generate B matrices
        real_part = np.random.random((dim, dim))  # Random values between 0 and 1
        imag_part = np.random.random((dim, dim))  # Random values between 0 and 1
        mat = real_part + 1j * imag_part
        mat = mat + mat.conjugate().T
        arr_B.append(mat)
    return arr_A, arr_B


def FPGPsd_facto(M, round_accuracy=10):
    """
    PSD factorization according to algorithm found in:
    Algorithms for Positive Semidefinite Factorization https://arxiv.org/pdf/1707.07953
    Page 5 Algorithm 3

    Parameters
    ------------
    M : list
        matrix we want to find psd-factorization for
    round_accuracy : int
        amount of decimals when rounding the values of the matrices (default=10)
    returns
    ----------------
    list
        two lists of factors
    """

    delta = 10
    M = np.array(M)
    dim = solve(M, print_steps=2)  # All the possible PSD-ranks
    for iter in range(len(dim)):  # Iterate for every possilbe PSD-rank
        arr_A, arr_B = generate_A_B_gradient(M, dim[iter])  # Intialize A and B matrices
        A = flatten(arr_A)  # Each factor is  flattened into a cloumn
        B = flatten(arr_B)
        scaling = sum(sum((M @ B.conjugate().T) * A.conjugate().T)) / sum(
            sum((B @ B.conjugate().T) * (A @ A.conjugate().T))
        )  # Scalar used to sacle A
        A = A * scaling  # Scale A
        for iter2 in range(2500):
            AX = B @ M.conjugate().T
            AAt = B @ B.conjugate().T
            A = faststepgrad(A, AX, AAt, delta)  # Optimize A
            AX = A @ M
            AAt = A @ A.conjugate().T
            B = faststepgrad(B, AX, AAt, delta)  # Optimize B
        arr_A = unflatten(A, dim[iter])  # Each column is its own matrix
        arr_B = unflatten(B, dim[iter])
        X = np.zeros(
            (M.shape[0], M.shape[1])
        )  # Calculate the matrix formed by the factros
        for i in range(len(arr_A)):
            for j in range(len(arr_B)):
                X[i, j] = np.trace(np.dot(arr_A[i], arr_B[j]))
        print(
            "A eigenvalues and matrices"
        )  # Print factros from A and their eigenvalues
        for mat in arr_A:
            print(np.round(mat, round_accuracy))
            print(np.linalg.eigh(mat)[0])
            print("\n")
        print(
            "\n B eigenvalues and matrices"
        )  # Print factros from B and their eigenvalues
        for mat in arr_B:
            print(np.round(mat, round_accuracy))
            print(np.linalg.eigh(mat)[0])
            print("\n")

        print("Original matrix M:")
        print(np.round(M, round_accuracy))
        print("New matrix X formed from factors: X_ij = Tr(A_iB_j)")
        print(np.round(X, round_accuracy))
        print(
            f"Frobenius norm (M-X): {np.linalg.norm(M-X)} \n"
        )  # Norm determines how good of a factorization was found
    return arr_A, arr_B


def faststepgrad(B, AX, AAt, delta):
    """
    Optimize the factors according to Alorithm 3 and 4 in https://arxiv.org/pdf/1707.07953

    Parameters
    ------------
    B : list
        matrix we want to optimize
    AX : list
        Used in optimizing B
    AAt : list
        Used in optimizing B
    delta : int
        determines the amount of iterations


    Returns
    -------------
    list
        optimized matrix
    """
    k = int(np.sqrt(AX.shape[0]))  # Shape of PSD factors
    n = B.shape[1]  # B's columns
    L = max(np.linalg.eigvalsh(AAt))  # Lipschitz constant
    Y = B
    for i in range(1, max(1, delta * k)):
        V = B + ((i - 2) / (i + 1)) * (B - Y)
        Y = B
        B = V - gradient(V, AAt, AX) / L  # Calculate gradient
        for j in range(0, n):
            B[:, j] = projection(B[:, j], k)  # Projection ensures PSD conditions
    return B


def projection(X, k):
    """
    Projection into the cone of PSD matrices as defined in https://arxiv.org/pdf/1707.07953

    Parameters
    -------------
    X : list
        the matrix we want to project
    k : int
        dimensions of PSD factros

    Returns
    -----------
    list
        projected matrix
    """

    X = np.reshape(X, (k, k))  # Reshape column of B (X) into shape of PSD factors
    V, D = np.linalg.eigh(X)  # The eigenvalues and -vectors of X
    eigvals_positive = np.diag(
        np.maximum(V, 0)
    )  # Make a diagonal matrix from positive eigenvalues

    X = D @ eigvals_positive @ D.conjugate().T  # Spectral decomposition
    return X.flatten()


def gradient(B, AAt, AX):
    """
    gradient as defined in https://arxiv.org/pdf/1707.07953


    Parameters
    -------------
    B : list
        the matrix we are optimizing
    AAt: list
        used in calculating the gradient
    AX: list
        used in calculating the gradient


    Returns
    ---------
    list
        the gradient
    """
    return AAt @ B - AX


def flatten(arr):
    """
    Takes a set of matrices and flattens those matrices into the columns of a larger matrix


    Parameters
    --------------
    arr : list
        the set of matrices

    Returns
    -----------
    list
        flattened matrix
    """
    flattened_columns = [elem.flatten() for elem in arr]  # Flatten each matrix
    new_matrix = np.column_stack(
        flattened_columns
    )  # flattened matrices are now columns
    return new_matrix


def unflatten(new_matrix, dim):
    """
    Unflatten the flattened matrices

    Parameters
    ----------
    new_matrix : list
        matrix with flattened matrice as columns
    dims : int
        determines the shape of unflattened matrices


    Returns
    ------------------
    list
        a set of unflattened matrices
    """
    num_columns = new_matrix.shape[1]

    original_matrices = [
        new_matrix[:, i].reshape((dim, dim)) for i in range(num_columns)
    ]  # Unflatten the matrices

    return original_matrices


for matrix in test_matrices.matrices.keys():
    alternating_strategy(test_matrices.matrices[matrix])
    FPGPsd_facto(test_matrices.matrices[matrix])
