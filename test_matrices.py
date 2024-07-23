import numpy as np
import lower_bound
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
NP = ([[1,0,0,25],[0,1,0,144],[0,0,1,169],[1,1,1,0]])  #Positive semidefinite rank theorem 5.6 page 19
matrices.update({"NP":NP})
sqrt = ([[1,0,1],[0,1,4],[1,1,1]])     #Positive semidefinite rank definition 5.2. page 18
matrices.update({"sqrt":sqrt})
M = ([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])    #Positive semidefinite rank 2.2 page 6
matrices.update({"M":M})
A_tensor_A = np.kron(A,A)
matrices.update({"A_tensor_A":A_tensor_A})
A_tensor_B = np.kron(A,B)
matrices.update({"A_tensor_B":A_tensor_B})
A_tensor_C = np.kron(A,C)
matrices.update({"A_tensor_C":A_tensor_C})
A_tensor_D = np.kron(A,D)
matrices.update({"A_tensor_D":A_tensor_D})
A_tensor_D_3 = np.kron(A,D_3)
matrices.update({"A_tensor_D_3":A_tensor_D_3})
A_tensor_sqrt = np.kron(A,sqrt)
matrices.update({"A_tensor_sqrt":A_tensor_sqrt})
B_tensor_A = np.kron(B,A)
matrices.update({"B_tensor_A":B_tensor_A})
B_tensor_B = np.kron(B,B)
matrices.update({"B_tensor_B":B_tensor_B})
B_tensor_C = np.kron(B,C)
matrices.update({"B_tensor_C":B_tensor_C})
B_tensor_D = np.kron(B,D)
matrices.update({"B_tensor_D":B_tensor_D})
B_tensor_D_3 = np.kron(B,D_3)
matrices.update({"B_tensor_D_3":B_tensor_D_3})
B_tensor_sqrt = np.kron(B,sqrt)
matrices.update({"B_tensor_sqrt":B_tensor_sqrt})
C_tensor_A = np.kron(C,A)
matrices.update({"C_tensor_A":C_tensor_A})
C_tensor_B = np.kron(C,B)
matrices.update({"C_tensor_B":C_tensor_B})
C_tensor_C = np.kron(C,C)
matrices.update({"C_tensor_C":C_tensor_C})
C_tensor_D = np.kron(C,D)
matrices.update({"C_tensor_D":C_tensor_D})
C_tensor_D_3 = np.kron(C,D_3)
matrices.update({"C_tensor_D_3":C_tensor_D_3})
C_tensor_sqrt = np.kron(C,sqrt)
matrices.update({"C_tensor_sqrt":C_tensor_sqrt})
D_tensor_A = np.kron(D,A)
matrices.update({"D_tensor_A":D_tensor_A})
D_tensor_B = np.kron(D,B)
matrices.update({"D_tensor_B":D_tensor_B})
D_tensor_C = np.kron(D,C)
matrices.update({"D_tensor_C":D_tensor_C})
D_tensor_D = np.kron(D,D)
matrices.update({"D_tensor_D":D_tensor_D})
D_tensor_D_3 = np.kron(D,D_3)
matrices.update({"D_tensor_D_3":D_tensor_D_3})
D_tensor_sqrt = np.kron(D,sqrt)
matrices.update({"D_tensor_sqrt":D_tensor_sqrt})
sqrt_tensor_A = np.kron(sqrt,A)
matrices.update({"sqrt_tensor_A":sqrt_tensor_A})
sqrt_tensor_B = np.kron(sqrt,B)
matrices.update({"sqrt_tensor_B":sqrt_tensor_B})
sqrt_tensor_C = np.kron(sqrt,C)
matrices.update({"sqrt_tensor_C":sqrt_tensor_C})
sqrt_tensor_D = np.kron(sqrt,D)
matrices.update({"sqrt_tensor_D":sqrt_tensor_D})
sqrt_tensor_D_3 = np.kron(sqrt,D_3)
matrices.update({"sqrt_tensor_D_3":sqrt_tensor_D_3})
sqrt_tensor_sqrt = np.kron(sqrt,sqrt)
matrices.update({"sqrt_tensor_sqrt":sqrt_tensor_sqrt})

def random_matrices(n_matrices=10000, rows=3, cols=4, range_max = 11):
    matrices = []
    for i in range(n_matrices):
        mat = np.random.randint(0, range_max, size=(rows, cols))
        mat = mat / mat.sum(axis=1, keepdims=True)
        matrices.append(mat)
    return matrices
