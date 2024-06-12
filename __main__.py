import lower_bound as lb
import upper_bound as ub
import test_matrices

for matrix in test_matrices.matrices.keys():
    lower_bound = lb.rank_based_lower_bound(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using rank_based")
    lower_bound = lb.B4(test_matrices.matrices[matrix])
    print(f"PSD Rank Lower bound for {matrix}: {lower_bound} B4")
    upper_bound = ub.mindim_upper_bound(test_matrices.matrices[matrix])
    print(f"PSD Rank Upper bound for {matrix}: {upper_bound}")

    