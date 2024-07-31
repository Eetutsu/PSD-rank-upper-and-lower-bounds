import lower_bound
import upper_bound
import math


def solve(M, print_steps = 0, print_rounded = True, eps = 0.0000001):
    """uses all the lower and upper bounds to solve the bounds for PSD-rank

    Parameters
    -----------
    M : list
        the matrix we want to solve PSD-rank for
    print_steps : int
        determines what will be printed, if 0, prints each bound calculated, if 1 prints the bounds for psd-rank, if 2 prints only if psd-rank is solved (default 0)
    print_rounded : boolean
        determines wheter values will be printed with three decimals (default True)
    eps : float
        determines accuracy for math.isclose()
    """
    lbs = []
    ubs = []
    if print_steps == 0:
        print("Lower Bounds:")
    for func in lower_bound.l_bounds: #Calculate lower bound using each method
        res = func(M)
        if print_steps == 0:
            if print_rounded:
                print(f"{func.__name__}: {res :.3f}") #Print the lower bound rounded
            else:
                print(f"{func.__name__}: {res}")    #Print the lower bound
        lbs.append(res)
    if print_steps == 0:
        print("Upper Bounds: ")
    for func in upper_bound.u_bounds:   #Calculate upper bound using each method
        res = func(M)
        if print_steps == 0:
            if print_rounded:
                print(f"{func.__name__}: {res :.3f}")   #Print upper bound rounded
            else:
                print(f"{func.__name__}: {res}")    #Print upper bound
        ubs.append(res)
    lb_max = max(lbs)
    if(math.isclose((lb_max),min(ubs),rel_tol=eps) or math.ceil(lb_max) == min(ubs)):   #Check if psd-rank was "solved"
        print(f"PSD-rank is {min(ubs)} for matrix :")    #Print psd-rank
        for row in M:   
            if print_rounded:
                rounded_row = [round(x,2) for x in row]
                print(rounded_row)  #Print for which matrix psd-rank was found for with each element rounded
            else:
                print(row)  #Print for which matrix psd-rank was found for
    else:
        if print_steps <=1:
            print(f"PSD-rank is within bounds: {math.ceil(lb_max):.1f}<=rank_psd<={min(ubs):.1f}. For matrix: ")    #Print the bounds for psd-rank
            for row in M:   
                if print_rounded:
                    rounded_row = [round(x,2) for x in row]
                    print(rounded_row)  #Print for which matrix psd-rank was found for with each element rounded
                else:
                    print(row)  #Print for which matrix psd-rank was found for
    return lbs