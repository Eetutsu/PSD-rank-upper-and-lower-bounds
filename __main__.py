import test_matrices
import time
from summary import solve
import math


def random():
    lb_methods = [0, 0, 0, 0, 0, 0, 0, 0]
    tic = time.perf_counter()
    test_matrices.random_matrices(range_max=6)
    for matrix in test_matrices.random_matrices(rows=4, cols=4, n_matrices=100):
        i = solve(matrix, print_steps=0, print_rounded=False)
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
        lbs = solve(test_matrices.matrices[matrix], print_rounded=False)
        print("\n")
        stats(lbs, lb_methods_correct, lb_methods_max)
        print("\n")

    toc = time.perf_counter()
    print(f"Aikaa kului: {toc - tic:0.4f} sekuntia")


def __main__():
    not_random()


def stats(lbs, lb_methods_correct, lb_methods_max):

    i = lbs.index(max(lbs))
    lb_methods_max[i] += 1
    for j in range(len(lbs)):
        if math.ceil(lbs[j]) == math.ceil(max(lbs)):
            lb_methods_correct[j] += 1
    print(f"{lb_methods_max}\n {lb_methods_correct}")


if __name__ == "__main__":
    __main__()
