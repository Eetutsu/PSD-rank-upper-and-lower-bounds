import numpy as np

#Matrices found in a paper written by Teiko Heinosaari, Oskari Kerppo and Leevi Leppäjärvi
#Called "COMMUNICATION TASKS IN OPERATIONAL THEORIES" https://arxiv.org/pdf/2003.05264
#Also some others from https://arxiv.org/pdf/1407.4308 and https://arxiv.org/pdf/1407.4095

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
#NP = ([[1,0,0,25],[0,1,0,144],[0,0,1,169],[1,1,1,0]])
#matrices.update({"NP":NP})
#sqrt = ([[1,0,1],[0,1,4],[1,1,1]])
#matrices.update({"sqrt":sqrt})
A_tensor = (np.kron(A,A))
matrices.update({"A_tensor":A_tensor})
#M = ([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
#matrices.update({"M":M})
#A_2 = ([[0,0,1,2,2,1],[1,0,0,1,2,2],[2,1,0,0,1,2],[2,2,1,0,0,1],[1,2,2,1,0,0],[0,1,2,2,1,0]])
#matrices.update({"A_2":A_2})