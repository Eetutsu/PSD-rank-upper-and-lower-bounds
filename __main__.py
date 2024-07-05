import lower_bound as lb
import upper_bound as ub
import test_matrices
import B3
import time
import math



def random():
    tic = time.perf_counter()
    test_matrices.random_matrices()
    for matrix in test_matrices.matrices_random.keys():
        ubs = []
        lbs = []
        lower_bound1 = lb.B3_gradient(test_matrices.matrices_random[matrix])
        lbs.append(lower_bound1)
        lower_bound2 = lb.B3_newton(test_matrices.matrices_random[matrix])
        lbs.append(lower_bound2)
        lower_bound3 = lb.B1(test_matrices.matrices_random[matrix])
        lbs.append(lower_bound3)
        lower_bound4 = lb.B4(test_matrices.matrices_random[matrix])
        lbs.append(lower_bound4)
        lower_bound5 = lb.B4D(test_matrices.matrices_random[matrix])
        lbs.append(lower_bound5)
        upper_bound1 = ub.mindim_upper_bound(test_matrices.matrices_random[matrix])
        ubs.append(upper_bound1)
        upper_bound2 = ub.hadamard_sqrt_upper_bound(test_matrices.matrices_random[matrix])
        ubs.append(upper_bound2)
        for elem in lbs:
            if isinstance(elem,str):
                elem = 0
        maximum = math.ceil(max(lbs))
        minimum = min(ubs)
        if maximum == minimum:
            print(f"Lower bounds for {matrix}: {test_matrices.matrices_random[matrix]}:")
            print(f"PSD Rank Lower bound: {lower_bound1} using gradient method")
            print(f"PSD Rank Lower bound: {lower_bound2} using newton mehod")
            print(f"PSD Rank Lower bound: {lower_bound3} using B1")
            print(f"PSD Rank Lower bound: {lower_bound4} using B4")
            print(f"PSD Rank Lower bound: {lower_bound5} using B4D")
            print("Upper bounds:")
            print(f"PSD Rank Upper bound: {upper_bound1} using mindim")
            print(f"PSD Rank Upper bound: {upper_bound2} using hadamard sqrt\n")
        else: continue


    toc = time.perf_counter()
    print(f"Aikaa kului: {toc - tic:0.4f} sekuntia")

def not_random():
    tic = time.perf_counter()

    for matrix in test_matrices.matrices.keys():
        print("Lower bounds:")
        lower_bound = lb.B3_gradient(test_matrices.matrices[matrix])
        print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using gradient method")
        lower_bound = lb.B3_newton(test_matrices.matrices[matrix])
        print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using newton mehod")
        lower_bound = lb.B1(test_matrices.matrices[matrix])
        print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using B1")
        lower_bound = lb.B4(test_matrices.matrices[matrix])
        print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using B4")
        lower_bound = lb.B4D(test_matrices.matrices[matrix])
        print(f"PSD Rank Lower bound for {matrix}: {lower_bound} using B4D")
        upper_bound = ub.mindim_upper_bound(test_matrices.matrices[matrix])
        print("Upper bounds:")
        print(f"PSD Rank Upper bound for {matrix}: {upper_bound} using mindim")
        upper_bound = ub.hadamard_sqrt_upper_bound(test_matrices.matrices[matrix])
        print(f"PSD Rank Upper bound for {matrix}: {upper_bound} using hadamard sqrt\n")
    toc = time.perf_counter()
    print(f"Aikaa kului: {toc - tic:0.4f} sekuntia")


def __main__():
    not_random()
    #random()

if __name__ == "__main__":
    __main__()
