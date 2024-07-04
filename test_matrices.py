import numpy as np
#Matrices found in a paper written by Teiko Heinosaari, Oskari Kerppo and Leevi Leppäjärvi
#Called "COMMUNICATION TASKS IN OPERATIONAL THEORIES" https://arxiv.org/pdf/2003.05264
#Also some others from https://arxiv.org/pdf/1407.4308 and https://arxiv.org/pdf/1407.4095


def normalize_mat(M):
    for row in M:
        summa = sum(row)
        for i in range(len(row)):
            row[i] = row[i] / summa
    return M


matrices = {}

K_plus = ([[1/2,1/2,0,0],[1/2,0,1/2,0],[1/2,0,0,1/2],[0,1/2,0,1/2],[0,0,1/2,1/2]])
matrices.update({"K_plus":K_plus})
K = ([[1/2,1/2,0,0],[1/2,0,1/2,0],[0,1/2,0,1/2],[0,0,1/2,1/2]])
matrices.update({"K":K})
K_minus = ([[1/2,1/2,0,0],[1/2,0,1/2,0],[0,1/2,0,1/2]])
matrices.update({"K_minus":K_minus})
D_3 = ([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
matrices.update({"D_3":D_3})
A = ([[1,0,0],[1/2,1/2,0],[1/2,0,1/2]])
matrices.update({"A":A})
B = ([[2/3,1/3,0],[0,2/3,1/3],[1/3,0,2/3]])
matrices.update({"B":B})
C = ([[1,0,0],[0,1/2,1/2],[1/2,0,1/2]])
matrices.update({"C":C})
D = ([[1,0,0],[0,1/2,1/2],[0,0,1]])
matrices.update({"D":D})
NP = normalize_mat([[1,0,0,25],[0,1,0,144],[0,0,1,169],[1,1,1,0]])  #Positive semidefinite rank theorem 5.6 page 19
matrices.update({"NP":NP})
sqrt = normalize_mat([[1,0,1],[0,1,4],[1,1,1]])     #Positive semidefinite rank definition 5.2. page 18
matrices.update({"sqrt":sqrt})
M = normalize_mat([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])    #Positive semidefinite rank 2.2 page 6
matrices.update({"M":M})
A_2 = normalize_mat([[0,0,1,2,2,1],[1,0,0,1,2,2],[2,1,0,0,1,2],[2,2,1,0,0,1],[1,2,2,1,0,0],[0,1,2,2,1,0]])     #Some upper and lower bounds example 36 page 13
matrices.update({"A_2":A_2})
A_tensor = np.kron(A,A)
matrices.update({"A_tensor":A_tensor})
E_1 = [[1/2,1/2,0,0,0],[0,1/2,1/2,0,0],[0,0,1/2,1/2,0],[0,0,0,1/2,1/2],[1/3,1/3,1/3,0,0]]
matrices.update({"Eetun oma 1":E_1})
E_2 = [[1/5,1/5,1/5,1/5,1/5],[1/4,1/4,1/4,0,1/4],[1/3,0,1/3,0,1/3],[0,0,0,1/2,1/2],[0,1,0,0,0]]
matrices.update({"Eetun oma 2":E_2})
E_3 = [[1/2,1/2,0,0,0],[0,1/2,1/2,0,0],[0,0,1/2,1/2,0],[0,0,0,1/2,1/2]]
matrices.update({"Eetun oma 3":E_3})
E_4 = [[1,0,0],[1/2,0,1/2],[1/2,1/2,0]]
matrices.update({"Eetun oma 4":E_4})
E_5 = [[1,0,0],[1/2,0,1/2],[1/3,1/3,1/3],[2/3,0,1/3]]
matrices.update({"Eetun oma 5":E_5})
E_6 = [[1/6,2/6,3/6],[3/6,1/6,2/6],[2/6,3/6,1/6]]
matrices.update({"Eetun oma 6":E_6})

