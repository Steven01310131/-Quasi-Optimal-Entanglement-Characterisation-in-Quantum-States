import numpy as np 


# Inner product function of the given state 'psi' and the product state 'phi'.
def Inner_product_function(phi, psi):
    # Perform some operation on the matrices, e.g., element-wise addition
    result = np.abs(np.vdot(phi, psi))
    return result

# Perform a tensor product operation on the rows of the matrix to obtain a vector representation of state 'phi' in the computational basis.
def tensor_product_function(matrix):
    num_rows=matrix.shape[0]
    phi=[1]
    for i in range(num_rows):
        phi=np.kron(phi, matrix[i])
    return phi

def problem(num_qubits,psi_real_coef,psi_im_coef,phi_real_coef,phi_im_coef):
    psi_state =np.array([])
    for i,j in zip(psi_real_coef,psi_im_coef):
        psi_state=np.concatenate((psi_state, np.array([complex(i, j)])))
    magnitude = np.linalg.norm(psi_state)
    psi_state = psi_state/ magnitude
    # Initialize an empty matrix of size (number of qubits) x 2, where each row represents one qubit state, to store a separable (product) state 'phi'.
    matrix = np.zeros((num_qubits, 2), dtype=complex)

    for i in range(num_qubits):
        for j in range(2):
            matrix[i, j] = complex(phi_real_coef[2*i+j], phi_im_coef[2*i+j])
        matrix[i]=matrix[i]/np.linalg.norm(matrix[i])  
    phi_state=tensor_product_function(matrix)
    iner_product=Inner_product_function(phi_state, psi_state)
    return iner_product

num_qubits=2
psi_real_coef=[1,2,3,1000]
psi_im_coef=[1,20000,3,4]
phi_real_coef=[1,2,3,4]
phi_im_coef=[4,3,2,1]
print(problem(num_qubits,psi_real_coef,psi_im_coef,phi_real_coef,phi_im_coef))