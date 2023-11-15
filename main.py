import numpy as np 
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize
from skopt.space import Real


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

def q2_problem(coefficients):
    length = len(coefficients) // 4

    # Divide the values into four lists
    psi_real_coef = coefficients[:length]
    psi_im_coef = coefficients[length:2*length]
    phi_real_coef = coefficients[2*length:3*length]
    phi_im_coef = coefficients[3*length:]
    psi_state =np.array([])
    for i,j in zip(psi_real_coef,psi_im_coef):
        psi_state=np.concatenate((psi_state, np.array([complex(i, j)])))
    magnitude = np.linalg.norm(psi_state)
    psi_state = psi_state/ magnitude
    # Initialize an empty matrix of size (number of qubits) x 2, where each row represents one qubit state, to store a separable (product) state 'phi'.
    matrix = np.zeros((2, 2), dtype=complex)

    for i in range(2):
        for j in range(2):
            matrix[i, j] = complex(phi_real_coef[2*i+j], phi_im_coef[2*i+j])
        matrix[i]=matrix[i]/np.linalg.norm(matrix[i])  
    phi_state=tensor_product_function(matrix)
    iner_product=Inner_product_function(phi_state, psi_state)
    return iner_product

space = [Real(-10.0, 10.0, name=f'x{i+1}') for i in range(16)] 
result = gp_minimize(q2_problem, space,acq_func="PI", n_calls=10, random_state=42)
print("Best parameters:", result.x)
print("Maximum value found:", -result.fun)  # Negate the result to get the actual maximum value

