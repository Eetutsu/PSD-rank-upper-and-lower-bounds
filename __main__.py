import lower_bound as lb
import upper_bound as ub
import test_matrices
import B3
import time

tic = time.perf_counter()
for matrix in test_matrices.matrices.keys():
    lower_bound = lb.B3_gradient(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using gradient method")
    lower_bound = B3.B3_newton(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using newton mehod")
    lower_bound = lb.rank_based_lower_bound(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using rank sqrt")
    lower_bound = lb.B4(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using B4")
    lower_bound = lb.B4(test_matrices.matrices[matrix], True)
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using B4D")
    upper_bound = ub.mindim_upper_bound(test_matrices.matrices[matrix])
    print(f"PSD Rank Upper bound for {matrix}: {upper_bound} using mindim")
    upper_bound = ub.hadamard_sqrt_upper_bound(test_matrices.matrices[matrix])
    print(f"PSD Rank Upper bound for {matrix}: {upper_bound} using hadamard sqrt stochastic")
    upper_bound = ub.hadamard_sqrt_upper_bound(test_matrices.matrices[matrix], is_accurate=True)
    print(f"PSD Rank Upper bound for {matrix}: {upper_bound} using hadamard sqrt accurate \n")
toc = time.perf_counter()
print(f"Aikaa kului: {toc - tic:0.4f} sekuntia")

