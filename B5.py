import numpy as np
import test_matrices


def generate_q(M):
    """Generates a probability distribution q with random values"""
    q = []
    for i in range(len(M)):
        q.append(np.random.randint(1,21))
    return normalize(q)


def normalize(q_temp):
    """normalizes the entries of a vector to sum up to 1"""
    q = []
    for elem in q_temp:
        q_elem = abs(elem/sum(q_temp))
        q.append(q_elem)
    return q


def F(M_i,M_j):
    """calculates the fideilty beteween the ith and jth row"""
    fid_sum = 0
    for k in range(len(M_i)):
        fid_sum += np.sqrt(M_i[k]*M_j[k])
    return fid_sum


def B5(M):
    M = np.array(M)
    sum = 0
    q = generate_q(M)
    for row in M:
        num = 0
        for k in range(len(M)):
            num += q[k]*(row[k])
        den = 0
        for s in range(len(M)):
            for t in range(len(M)):
                den += q[s]*q[t]*F(M[s],M[t])**2
        den = np.sqrt(den)
        sum += num/den
    return sum
                
for matrix in test_matrices.matrices.keys():
    lower_bound = B5(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using B5")