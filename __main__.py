import lower_bound as lb
import upper_bound as ub
import test_matrices
import time

tic = time.perf_counter()
for matrix in test_matrices.matrices.keys():
    lower_bound = lb.rank_based_lower_bound(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using rank sqrt")
    lower_bound = lb.B4(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using B4")
    lower_bound = lb.B4(test_matrices.matrices[matrix], True)
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using B4D")
    lower_bound = lb.B3_gradient(test_matrices.matrices[matrix],lr_scaler=0.99)
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using fidelity \n")

toc = time.perf_counter()
print(f"Aikaa kului: {toc - tic:0.4f} sekuntia")

