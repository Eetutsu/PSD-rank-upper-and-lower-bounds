import lower_bound
import upper_bound
import math


def solve(M, printed = 0, round_print = True):
    lbs = []
    ubs = []
    if printed == 0:
        print("Lower Bounds:")
    for func in lower_bound.l_bounds:
        if printed == 0:
            if round_print:
                print(f"{func.__name__}: {func(M) :.3f}")
            else:
                print(f"{func.__name__}: {func(M)}")
        lbs.append(func(M))
    if printed == 0:
        print("Upper Bounds: ")
    for func in upper_bound.u_bounds:
        if printed == 0:
            if round_print:
                print(f"{func.__name__}: {func(M) :.3f}")
            else:
                print(f"{func.__name__}: {func(M)}")
        ubs.append(func(M))
    if(math.ceil(max(lbs)) == min(ubs)):
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
            print(f"PSD-rank is within bounds: {math.ceil(max(lbs)):.1f}<=rank_psd<={min(ubs):.1f}. For matrix: ")
            for row in M:
                if round_print:
                    rounded_row = [round(x,2) for x in row]
                    print(rounded_row)
                else:
                    print(row)
            print("\n")

#for matrix in test_matrices.matrices.values():
#    solve(matrix)