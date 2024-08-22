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
    arr_A = []  # Array of matrices A
    arr_B = []  # Array of matrices B
    for i in range(M.shape[0]):  # Generate A matrices
        real_part = np.random.random((dim, dim))  # Random values between 0 and 1
        imag_part = np.random.random((dim, dim))  # Random values between 0 and 1
        mat = real_part + 1j * imag_part
        mat = mat + mat.conjugate().T
        arr_A.append(mat)  # Ensures the matrcies are symmetric
    for j in range(M.shape[1]):  # Generate B matrices
        real_part = np.random.random((dim, dim))  # Random values between 0 and 1
        imag_part = np.random.random((dim, dim))  # Random values between 0 and 1
        mat = real_part + 1j * imag_part
        mat = mat + mat.conjugate().T
        arr_B.append(mat)
    return arr_A, arr_B


def accelerated_gradinet(M):
    delta = 10
    M = np.array(M)
    dim = solve(M, print_steps=2)  # All the possible PSD-ranks
    for iter in range(len(dim)):  # Iterate for every possilbe PSD-rank
        arr_A, arr_B = generate_A_B(M, dim[iter])  # Intialize A and B matrices
        A = flatten(arr_A)
        B = flatten(arr_B)
        scaling = sum(sum((M @ B.conjugate().T) * A.conjugate().T)) / sum(
            sum((B @ B.conjugate().T) * (A @ A.conjugate().T))
        )
        A = A * scaling
        for iter2 in range(2500):
            AX = B @ M.conjugate().T
            AAt = B @ B.conjugate().T
            A = faststepgrad(A, AX, AAt, delta)
            AX = A @ M
            AAt = A @ A.conjugate().T
            B = faststepgrad(B, AX, AAt, delta)
        arr_A = unflatten(A, dim[iter])
        arr_B = unflatten(B, dim[iter])
        X = np.zeros((M.shape[0], M.shape[1]))
        for i in range(len(arr_A)):
            for j in range(len(arr_B)):
                X[i, j] = np.trace(np.dot(arr_A[i], arr_B[j]))
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
    return arr_A, arr_B


def faststepgrad(B, AX, AAt, delta):
    k = int(np.sqrt(AX.shape[0]))
    n = B.shape[1]
    L = max(np.linalg.eigvalsh(AAt))
    Y = B
    for i in range(1, max(1, delta * k)):
        V = B + ((i - 2) / (i + 1)) * (B - Y)
        Y = B
        B = V - gradient(V, AAt, AX) / L
        for j in range(0, n):
            B[:, j] = projection(B[:, j], k)
    return B


def projection(X, k):
    X = np.reshape(X, (k, k))
    V, D = np.linalg.eigh(X)
    eigvals_positive = np.diag(np.maximum(V, 0))

    X = D @ eigvals_positive @ D.conjugate().T
    return X.flatten()


def gradient(B, AAt, AX):
    return AAt @ B - AX


def flatten(arr):
    flattened_columns = [elem.flatten() for elem in arr]
    new_matrix = np.column_stack(flattened_columns)
    return new_matrix


def unflatten(new_matrix, dim):
    num_columns = new_matrix.shape[1]

    original_matrices = [
        new_matrix[:, i].reshape((dim, dim)) for i in range(num_columns)
    ]

    return original_matrices


for matrix in test_matrices.matrices.values():
    accelerated_gradinet(matrix)
    print("\n")
