import lower_bound as lb
import test_matrices

for matrix in test_matrices.matrices.keys():
    lower_bound = lb.rank_based_lower_bound(test_matrices.matrices[matrix])
    print(f"Lower bound for {matrix}: {lower_bound}")