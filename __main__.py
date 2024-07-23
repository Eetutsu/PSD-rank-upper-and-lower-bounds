import lower_bound as lb
import upper_bound as ub
import test_matrices
import time
from summary import solve
import summary

def random():
    lb_methods = [0, 0, 0, 0, 0, 0, 0, 0]
    tic = time.perf_counter()
    test_matrices.random_matrices(range_max=6)
    for matrix in test_matrices.random_matrices(rows=4,cols=4,n_matrices=100):
        i = solve(matrix,printed=0,round_print=False)
        lb_methods[i] = lb_methods[i] + 1
        print(lb_methods)

    toc = time.perf_counter()
    print(f"Aikaa kului: {toc - tic:0.4f} sekuntia")

def not_random():
    tic = time.perf_counter()
    lb_methods_max = [0, 0, 0, 0, 0, 0, 0, 0]
    for matrix in test_matrices.matrices.values():
        i = solve(matrix, round_print= False)
        lb_methods_max[i] = lb_methods_max[i] + 1
        print(lb_methods_max)


    toc = time.perf_counter()
    print(f"Aikaa kului: {toc - tic:0.4f} sekuntia")


def __main__():
    not_random()
    #random()
    #solve([[0.0, 1.0, 0.0, 0.0],[0.41, 0.21, 0.21, 0.17],[0.41, 0.21, 0.21, 0.17],[0.21, 0.08, 0.29, 0.42]])
    #solve([[0.47, 0.32, 0.21, 0.0],[0.4, 0.27, 0.33, 0.0],[0.4, 0.12, 0.2, 0.28],[0.23, 0.15, 0.62, 0.0]])
    #solve([[0.47619048, 0.23809524, 0, 0.28571429,],[0.875, 0,    0,   0.125],[0.3, 0.26666667,0.16666667,0.26666667],[0.88888889, 0, 0, 0.11111111],[0.38095238, 0.14285714, 0, 0.47619048]])
    #solve(test_matrices.D_tensor_D_3)
    #solve([[0, 1, 1, 1, 1, 1],[1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1],[1, 1, 1, 0, 1, 1],[1, 1, 1, 1, 0, 1],[1, 1, 1, 1, 1, 0]])
    #solve([[1, 1, 0, 0],[0, 1, 1, 0], [0, 0, 1, 1],[1, 0, 0, 1]])
    #solve([[1, 1, 0, 0, 1],[1, 1, 0, 0, 1],[0, 0, 1, 1, 0],[0, 0, 1, 1, 0],[1, 1, 0, 0, 1]])
    #solve([[1, 1, 1, 0],[1, 1, 1, 0],[1, 1, 1, 0],[0, 0, 0, 1]])
    #solve([[1, 0, 1],[0, 1, 1],[1, 1, 1]])
    #solve([[1, 1],[1, 1]])
    #solve([[1, 0],[0, 1]])


if __name__ == "__main__":
    __main__()
