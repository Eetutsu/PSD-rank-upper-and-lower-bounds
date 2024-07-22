import lower_bound
import upper_bound
import math


def solve(M, printed = 0, round_print = True, eps = 0.0000001):
    lbs = []
    ubs = []
    if printed == 0:
        print("Lower Bounds:")
    for func in lower_bound.l_bounds:
        res = func(M)
        if printed == 0:
            if round_print:
                print(f"{func.__name__}: {res :.3f}")
            else:
                print(f"{func.__name__}: {res}")
        lbs.append(res)
    if printed == 0:
        print("Upper Bounds: ")
    for func in upper_bound.u_bounds:
        res = func(M)
        if printed == 0:
            if round_print:
                print(f"{func.__name__}: {res :.3f}")
            else:
                print(f"{func.__name__}: {res}")
        ubs.append(res)
    lb_max = max(lbs)
    if(math.isclose((lb_max),min(ubs),rel_tol=eps)):
        print(f"PSD-rank is {min(ubs)} for matrix:")
        if round_print:
            for row in M:
                rounded_row = [round(x,2) for x in row]
                print(rounded_row)
        else:
            for row in M:
                print(row)
        print("\n")
    else:
        if printed <=1:
            print(f"PSD-rank is within bounds: {math.ceil(lb_max):.1f}<=rank_psd<={min(ubs):.1f}. For matrix: ")
            for row in M:
                if round_print:
                    rounded_row = [round(x,2) for x in row]
                    print(rounded_row)
                else:
                    print(row)
            print("\n")

#for matrix in test_matrices.matrices.values():
#    solve(matrix)