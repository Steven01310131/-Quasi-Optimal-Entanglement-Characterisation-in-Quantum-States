import numpy as np 

l=[1,2,3,4]
l2=[1,2,3,4]
psi_state =np.array([])
for i,j in zip(l,l2):
    psi_state=np.concatenate((psi_state, np.array([complex(i, j)])))
l=[]
print(psi_state)
matrix = np.zeros((2, 2), dtype=complex)
l1=[1,2,3,4]
l2=[1,2,3,4]
for i in range(2):
    for j in range(2):
        matrix[i, j] = complex(l1[2*i+j], l2[2*i+j])
    matrix[i]=matrix[i]/np.linalg.norm(matrix[i])  
print(matrix)