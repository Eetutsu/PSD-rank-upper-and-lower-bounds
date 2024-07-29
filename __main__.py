import lower_bound as lb
import upper_bound as ub
import test_matrices
import time
from summary import solve
import math

def random():
    lb_methods = [0, 0, 0, 0, 0, 0, 0, 0]
    tic = time.perf_counter()
    test_matrices.random_matrices(range_max=6)
    for matrix in test_matrices.random_matrices(rows=4,cols=4,n_matrices=100):
        i = solve(matrix,print_steps=0,print_rounded=False)
        lb_methods[i] = lb_methods[i] + 1
        print(lb_methods)

    toc = time.perf_counter()
    print(f"Aikaa kului: {toc - tic:0.4f} sekuntia")

def not_random():
    lb_methods_max = [0, 0, 0, 0, 0, 0, 0, 0]
    lb_methods_correct = [0, 0, 0, 0, 0, 0, 0, 0]
    tic = time.perf_counter()
    for matrix in test_matrices.matrices.keys():
        print(f"PSD-rank solved for: {matrix}")
        lbs = solve(test_matrices.matrices[matrix], print_rounded= False)
        stats(lbs,lb_methods_correct,lb_methods_max)


    toc = time.perf_counter()
    print(f"Aikaa kului: {toc - tic:0.4f} sekuntia")


def __main__():
    not_random()
    #random()
    #solve([[0.0, 1.0, 0.0, 0.0],[0.41, 0.21, 0.21, 0.17],[0.41, 0.21, 0.21, 0.17],[0.21, 0.08, 0.29, 0.42]])
    #solve([[0.47619048, 0.23809524, 0, 0.28571429,],[0.875, 0,    0,   0.125],[0.3, 0.26666667,0.16666667,0.26666667],[0.88888889, 0, 0, 0.11111111],[0.38095238, 0.14285714, 0, 0.47619048]])
    #solve(test_matrices.D_tensor_D_3)
    #solve([[0, 1, 1, 1, 1, 1],[1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1],[1, 1, 1, 0, 1, 1],[1, 1, 1, 1, 0, 1],[1, 1, 1, 1, 1, 0]])
    #solve([[1, 1, 0, 0],[0, 1, 1, 0], [0, 0, 1, 1],[1, 0, 0, 1]])
    #solve([[1, 1, 0, 0, 1],[1, 1, 0, 0, 1],[0, 0, 1, 1, 0],[0, 0, 1, 1, 0],[1, 1, 0, 0, 1]])
    #solve([[1, 1, 1, 0],[1, 1, 1, 0],[1, 1, 1, 0],[0, 0, 0, 1]])
    #solve([[1, 0, 1],[0, 1, 1],[1, 1, 1]])
    #solve([[1, 1],[1, 1]])
    #solve([[1, 0],[0, 1]])


def stats(lbs,lb_methods_correct,lb_methods_max):
    
    i   = lbs.index(max(lbs))
    lb_methods_max[i] += 1
    for j  in range(len(lbs)):
        if math.ceil(lbs[j]) == math.ceil(max(lbs)):
            lb_methods_correct[j] +=1
    print(f"{lb_methods_max}\n {lb_methods_correct}\n")

if __name__ == "__main__":
    __main__()
