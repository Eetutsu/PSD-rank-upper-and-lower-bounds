import numpy as np
import math
from numpy.linalg import matrix_rank


def normalize_mat(M):
    M = np.array(M, dtype=float)
    row_sums = M.sum(axis=1, keepdims=True)
    return M / row_sums


def is_stochastic(M):
    M = np.array(M)
    return np.allclose(M.sum(axis=1), 1, rtol=1e-6)


def B1(M):
    return 0.5 * (np.sqrt(1 + 8 * matrix_rank(M)) - 1)


def B4(M, is_D=False):
    M = np.array(M)
    if not is_stochastic(M) and not is_D:
        return 0
    return np.sum(np.max(M.T, axis=1))


def F(M_i, M_j):
    return np.sum(np.sqrt(M_i * M_j))


def normalize(v):
    v = np.abs(v)
    return v / np.sum(v)


def generate_q(n):
    q = np.random.randint(1, 21, size=n)
    return normalize(q)


def grad_vec_min_B3(M, q, F_cache):
    gradient = []
    for i in range(len(M)):
        gradient.append(np.sum([2 * q[j] + F_cache[i, j] ** 2 for j in range(len(M))]))
    return gradient


def B3_gradient(M, lr=0.001, max_iter=10000, lr_scaler=0.95, eps=1e-5):
    M = np.array(M)
    if not is_stochastic(M):
        return 0
    F_cache = np.array([[F(M[i], M[j]) for j in range(len(M))] for i in range(len(M))])
    res_log = []
    for _ in range(100):
        q = generate_q(len(M))
        for _ in range(max_iter):
            gradient = grad_vec_min_B3(M, q, F_cache)
            q = np.clip(q - lr * np.array(gradient), 0, None)
            q = normalize(q)
            lr *= lr_scaler
            if np.max(np.abs(gradient)) < eps:
                break
        fidelity_sum = np.sum([q[i] * q[j] * F_cache[i, j] ** 2 for i in range(len(M)) for j in range(len(M))])
        res_log.append(1 / fidelity_sum)
    return max(res_log)


def optimize_with_D(M, score_fn, steps=1000, rounds=10):
    M = np.array(M, dtype=float)
    if not is_stochastic(M):
        return 0
    D = np.diag(np.random.randint(1, 11, size=M.shape[0]))
    best = 0
    for _ in range(rounds):
        for _ in range(steps):
            P0 = normalize_mat(D @ M)
            D += np.diag(np.random.rand(M.shape[0]) * 0.01)
            P1 = normalize_mat(D @ M)
            B0 = score_fn(P0)
            B1_ = score_fn(P1)
            grad = (B1_ - B0) / 0.001
            D += np.diag(np.full(M.shape[0], grad * 0.001))
        D *= 0.95
        best = max(best, B1_)
    return best


def B4D(M):
    return optimize_with_D(M, lambda P: B4(P, is_D=True))


def B3_newton(M, lr=0.01, eps=1e-6, lr_scaler=0.95):
    M = np.array(M)
    if not is_stochastic(M):
        return 0
    F_cache = np.array([[F(M[i], M[j]) for j in range(len(M))] for i in range(len(M))])

    def newton_iter(q):
        n = len(q)
        Hessian = np.array([[2 * F_cache[i, j] ** 2 for j in range(n)] for i in range(n)])
        grad = grad_vec_min_B3(M, q, F_cache)
        try:
            return np.linalg.solve(Hessian, grad)
        except np.linalg.LinAlgError:
            return None

    res_log = []
    for _ in range(100):
        q = generate_q(len(M))
        for _ in range(10000):
            step = newton_iter(q)
            if step is None:
                return 0
            q = np.clip(q - lr * step, 0, None)
            q = normalize(q)
            lr *= lr_scaler
            if np.max(np.abs(step)) < eps:
                break
        fidelity_sum = np.sum([q[i] * q[j] * F_cache[i, j] ** 2 for i in range(len(M)) for j in range(len(M))])
        res_log.append(1 / fidelity_sum)
    return max(res_log)


def B5(M):
    M = np.array(M)
    if not is_stochastic(M):
        return 0
    F_cache = np.array([[F(M[i], M[j]) for j in range(len(M))] for i in range(len(M))])
    max_fid = 0
    for _ in range(100):
        q = generate_q(len(M))
        for _ in range(1000):
            gradient = grad_vec_min_B3(M, q, F_cache)
            q -= 0.001 * np.array(gradient)
            q = np.clip(q, 0, None)
            q = normalize(q)
        val = 0
        for i in range(len(M)):
            for j in range(len(M)):
                val += (q[i] + q[j]) * F_cache[i, j]
        max_fid = max(max_fid, val)
    return max_fid


def B5D(M):
    return optimize_with_D(M, B5)


def B3D_gradient(M):
    return optimize_with_D(M, B3_gradient)
