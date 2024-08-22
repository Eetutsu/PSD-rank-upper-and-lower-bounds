import numpy as np
import math
import test_matrices


def B5_gradient(M, eps=0.000001, lr=0.001, lr_scaler=0.99):
    """Calculates a lower bound on the PSD-rank for a row stochastic matrix using
    gradient descent.

    lower bound calculated using the method found in: https://arxiv.org/pdf/1407.4308
    Page 10 Definition 28

    Parameters
    ----------
    M : list
        the matrix we want to find a PSD-rank lower bound for
    eps : float
        used to approximate the derivate (default 0.001)
    lr : float
        the learning rate used when iterating (default 0.01)
    lr_scaler : float
        scales the learning rate after each iteration (default 0.95)

    Returns
    ----------
    int
        if not a row sotchastic matrix
    float
        the lower bound
    """

    if not is_stochastic(M):
        return 0  # Method can only be used if matrix is stochastic
    M = np.array(M)
    res_log = []
    q_log = [0, 0]
    for iter1 in range(10):
        max_log = []
        for i in range(len(M.T)):
            lr = 0.001
            q = generate_q(M)
            q_log[0] = q
            gradient = grad_vec_min(M, q, i)
            B5_log = []
            for iter2 in range(1000):
                for j in range(len(q)):
                    q[j] = q[j] + lr * gradient[j]
                    # Keeps q within bounds
                    if q[j] < 0:
                        q[j] = 0
                # Update the step size (learning rate)
                lr = lr * lr_scaler
                # Normalize q so that its entries sum up to one
                q = normalize(q)
                # log vector q_(i+1)
                q_log[1] = q
                q_temp = np.array([q_log[0][i] - q_log[1][i] for i in range(len(q))])
                # Check if change is so small that iterating further is not sensible
                if max(q_temp) < eps:
                    B5_log.append(calc_B5(M, q, i))
                    break
                # vector q_i+1 is now q_i
                q_log[0] = q_log[1]
                # Calculate the the gradient vector in the point q_i
                gradient = grad_vec_min(M, q, i)

                B5_log.append(calc_B5(M, q, i))
            max_log.append(max(B5_log))
        res_log.append(sum(max_log))
        print(f"B5_gradient gradient: {gradient}")

    return max(res_log)


def is_stochastic(M):
    """Checks wheter a matrix is row stochastic or not

    Parameters
    ----------
    M : list
        the matrix we want to check if it is row stochastic

    Returns
    ----------
    boolean
        returns True if it is row stochastic, if not, returns False
    """

    rowsum = 0
    for row in M:
        for i in row:
            rowsum += i
        if not math.isclose(1, rowsum, rel_tol=1e-06):
            return False
        else:
            rowsum = 0
    return True


def generate_q(M):
    """Generates a probability distribution q with random values

    Parameters
    ----------
    M : list
      determines the size of q

    Returns
    ----------
    list
        probability distribution q
    """

    q = []
    for i in range(len(M)):
        q.append(np.random.randint(1, 21))
    return normalize(q)


def normalize(q_temp):
    """normalizes the entries of a vector to sum up to 1

    Parameters
    ----------
    q_temp : list
      the vector we want to normalize

    Returns
    ----------
    list
      a vector with entries that sum up to one (probability distribution)
    """

    q = []
    for elem in q_temp:
        q_elem = abs(elem / sum(q_temp))
        q.append(q_elem)
    return q


def grad_vec_min(M, q, i):
    """calculates the gradient vector for B3 in the point q

    Parameters
    ----------
    M : list
      the matrix used to calculate the fidelity
    q : list
      the vector that determines the point in which the gradient is calculated

    Returns
    ----------
    list
        the gradient vector
    """

    M = np.array(M)
    sum = 0
    gradient = []
    denominator = 0
    num1 = 0
    num2 = 0
    for s in range(len(q)):
        for t in range(len(q)):
            denominator = denominator + q[s] * q[t] * F(M[s], M[t]) ** 2
    for x in range(len(q)):
        num1 = M[x, i]
        num2 = q[x] * F(M[x], M[x]) ** 2
        for s in range(len(q)):
            num2 = num2 + q[s] * F(M[x], M[s]) ** 2
        temp = 0
        for k in range(len(q)):
            temp += q[k] * M[k, i]
        num2 *= temp
        sum = num1 / np.sqrt(denominator) - (num2 / denominator ** (3 / 2))
        gradient.append(sum)
    return gradient


def F(M_i, M_j):
    """calculates the fideilty beteween the ith and jth row as defined in:
    https://link.springer.com/article/10.1007/s10107-016-1052-0  Page 499

    Parameters
    ----------
    M_i : list
      ith row of the matrix M
    M_j : list
      jth row of the matrix M

    Returns
    ----------
    float
        the fidelity between M_i and M_j
    """

    fid_sum = 0
    for k in range(len(M_i)):
        fid_sum += np.sqrt(M_i[k] * M_j[k])
    return fid_sum


def calc_B5(M, q, i):
    """Calculates B5(M) according to q

    Parameters
    ----------
    M   : list
        the matrix we want to calculate B5(M) for
    q : list
        used to calculate denominator
    i : int
        index used for matrix M

    Returns
    --------
    float
        the calulated value for fraction part of B5(M)
    """

    summa1 = 0
    for s in range(len(q)):
        for t in range(len(q)):
            summa1 = summa1 + q[s] * q[t] * F(M[s], M[t]) ** 2
    k = 0
    summa2 = 0
    while k < len(q):
        summa2 = summa2 + q[k] * M[k][i]
        k += 1
    return summa2 / np.sqrt(summa1)


def B5(M, eps=0.0001, lr=0.001, lr_scaler=0.95):
    """Calculates a lower bound on the PSD-rank for a row stochastic matrix using
    gradient descent.

    lower bound calculated using the method found in: https://arxiv.org/pdf/1407.4308
    Page 10 Definition 28

    Parameters
    ----------
    M : list
        the matrix we want to find a PSD-rank lower bound for
    eps : float
        used to approximate the derivate (default 0.001)
    lr : float
        the learning rate used when iterating (default 0.01)
    lr_scaler : float
        scales the learning rate after each iteration (default 0.95)

    Returns
    ----------
    int
        if not a row sotchastic matrix
    float
        the lower bound
    """

    if not is_stochastic(M):
        return 0  # Method can only be used if matrix is stochastic
    M = np.array(M)
    sums = []
    summed = []
    grad = []
    P_k_log = []

    for iter in range(3):
        for i in range((M.shape[1])):
            lr = 0.001  # reset learning rate after maximizing q^(i)
            eps = 0.0001
            q = generate_q(M)  # Generate random probability distribution
            for iter in range(1000):
                grad.clear()
                for k in range((len(q))):
                    P_k_log.append(
                        calc_B5(M, q, i)
                    )  # Calculate fraction part of B5 with vector q_0
                    q[k] = q[k] + eps
                    P_k_log.append(
                        calc_B5(M, q, i)
                    )  # Calculate fraction part of B5 with vector q_1
                    grad.append(
                        (P_k_log[-1] - P_k_log[-2]) / eps
                    )  # Approximate the derivate and add it to the gradient vector
                eps = eps * 0.95
                for x in range(len(q)):
                    q[x] = (
                        q[x] + lr * grad[x]
                    )  # update q according to the gradient vector
                q = normalize(q)
                if all(math.isclose(v, 0, rel_tol=1e-04) for v in grad):
                    break
                lr = lr * lr_scaler
            sums.append(max(P_k_log))
        summed.append(sum(sums))  # Calculate B5(M)
        sums.clear()
        P_k_log.clear()
        print(f"B5 gradient: {grad}")

    return max(summed)  # Return the best calculated lower bound


for matrix in test_matrices.matrices.keys():
    print((B5(test_matrices.matrices[matrix])))
