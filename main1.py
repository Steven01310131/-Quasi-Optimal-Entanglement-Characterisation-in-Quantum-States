import numpy as np 
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize
from skopt.space import Real,Integer
import cmath 


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

def sph2cart(azimuth,elevation,r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z

#Main function
# l is a list with the given coeffiicients of the given psi 
# N the discretization space 
def main(l,N):
    k = np.arange(1, N+1)
    h = -1 + 2/(N-1) * (k-1)
    phi = np.arccos(h) - np.pi/2
    theta = np.zeros(N)
    theta[0] = 0

    for j in range(1, N-1):
        theta[j] = (theta[j-1] + 3.6/np.sqrt(N) * 1/np.sqrt(1 - h[j]**2)) % (2*np.pi)- np.pi
    theta[N-1] = 0
    def q2_problem(coefficients):
        # Divide the values into four lists
        psi_real_coef = l[:4]
        psi_im_coef = l[4:2*4]
        phi_1=phi[coefficients[0]]
        theta_1=theta[coefficients[1]]
        phi_2=phi[coefficients[2]]
        theta_2=theta[coefficients[3]]
        
        a_1 = cmath.cos(complex(0 , theta_1 / 2))
        a_2 = cmath.cos(complex(0 , theta_2 / 2))
        beta_1 = np.exp(complex(0 , phi_1)) *  cmath.sin(complex(0 , theta_1 / 2))
        beta_2 = np.exp(complex(0 , phi_2)) *  cmath.sin(complex(0 , theta_2 / 2))
        psi_state =np.array([])
        for i,j in zip(psi_real_coef,psi_im_coef):
            psi_state=np.concatenate((psi_state, np.array([complex(i, j)])))
        magnitude = np.linalg.norm(psi_state)
        psi_state = psi_state/ magnitude
        # Initialize an empty matrix of size (number of qubits) x 2, where each row represents one qubit state, to store a separable (product) state 'phi'.
        inner_product = a_1 * a_2 *  
        return -iner_product

    space = [Integer(0, N, name=f'x{i+1}') for i in range(4)]
    print(space)
    result = gp_minimize(q2_problem, space,acq_func="PI", n_calls=50, random_state=42)
    print("Best parameters:", result.x)
    print("Maximum value found:", -result.fun)  # Negate the result to get the actual maximum value

main([1,2,3,4,1,2,3,4],500)