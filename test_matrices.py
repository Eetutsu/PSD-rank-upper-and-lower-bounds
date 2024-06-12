import numpy as np

#Matrices found in a paper written by Teiko Heinosaari, Oskari Kerppo and Leevi Leppäjärvi
#Called "COMMUNICATION TASKS IN OPERATIONAL THEORIES" https://arxiv.org/pdf/2003.05264

matrices = {}

K_plus = np.matrix([[1/2,1/2,0,0],[1/2,0,1/2,0],[1/2,0,0,1/2],[0,1/2,0,1/2],[0,0,1/2,1/2]])
matrices.update({"K_plus":K_plus})
K = np.matrix([[1/2,1/2,0,0],[1/2,0,1/2,0],[0,1/2,0,1/2],[0,0,1/2,1/2]])
matrices.update({"K":K})
K_minus = np.matrix([[1/2,1/2,0,0],[1/2,0,1/2,0],[0,1/2,0,1/2]])
matrices.update({"K_minus":K_minus})
D_3 = np.matrix([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
matrices.update({"D_3":D_3})
A = np.matrix([[1,0,0],[1/2,1/2,0],[1/2,0,1/2]])
matrices.update({"A":A})
B = np.matrix([[2/3,1/3,0],[0,2/3,1/3],[1/3,0,2/3]])
matrices.update({"B":B})
C = np.matrix([[1,0,0],[0,1/2,1/2],[1/2,0,1/2]])
matrices.update({"C":C})
D = np.matrix([[1,0,0],[0,1/2,1/2],[0,0,1]])
matrices.update({"D":D})