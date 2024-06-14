import lower_bound as lb
import upper_bound as ub
import test_matrices

for matrix in test_matrices.matrices.keys():
    lower_bound = lb.rank_based_lower_bound(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using rank sqrt")
    lower_bound = lb.B4(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using B4")
    lower_bound = lb.B4(test_matrices.matrices[matrix], True)
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using B4")
    lower_bound = lb.B3(test_matrices.matrices[matrix])
    print(f"fidelity {matrix}: {lower_bound} \n")
    