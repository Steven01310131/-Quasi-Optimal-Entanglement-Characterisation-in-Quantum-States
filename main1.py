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
    # The coefficients are a list of integers which points to a location of the
    # phi and theta sample list of the polar and azimouthian cooridnates 
    def q2_problem(coefficients):
        # Divide the real part and imaginary part coefficients
        psi_real_coef = l[:4]
        psi_im_coef = l[4:2*4]

        #sample the polar and azimouthian coordinates from the theta and phi value list
        phi_1=phi[int(coefficients[0])]
        theta_1=theta[int(coefficients[1])]
        phi_2=phi[int(coefficients[2])]
        theta_2=theta[int(coefficients[3])]

        a_1 = cmath.cos(complex(0 , theta_1 / 2))
        a_2 = cmath.cos(complex(0 , theta_2 / 2))
        beta_1 = np.exp(complex(0 , phi_1)) *  cmath.sin(complex(0 , theta_1 / 2))
        beta_2 = np.exp(complex(0 , phi_2)) *  cmath.sin(complex(0 , theta_2 / 2))
        psi_state =np.array([])
        
        # Given vector phi
        for i,j in zip(psi_real_coef,psi_im_coef):
            psi_state=np.concatenate((psi_state, np.array([complex(i, j)])))
        magnitude = np.linalg.norm(psi_state)
        psi_state = psi_state/ magnitude

        inner_product = (np.conj(a_1 * a_2 )) * psi_state[0] + (np.conj(a_1 * beta_2 )) * psi_state[1] + (np.conj(beta_1 * a_2 )) * psi_state[2] + (np.conj(beta_1 * beta_2 )) * psi_state[3] 
        #Return the negative becauce we want to maximize
        return -np.abs(inner_product)
    
    #TODO fix the bug with the real since when I change the space from Real to int the program 
    # fails since somewhere the library uses np.int which is deprecated
    
    space = [Real(0, N-1, name=f'x{i+1}') for i in range(4)]
    result = gp_minimize(q2_problem, space,acq_func="PI", n_calls=50, random_state=42)
    # The best parameters are the indexes which points to the best phi and theta 
    print("Best parameters:")
    print(f"Ph1:{phi[int(result.x[0])]}")
    print(f"theta1:{theta[int(result.x[1])]}")
    print(f"Ph2:{phi[int(result.x[2])]}")
    print(f"theta2:{theta[int(result.x[3])]}")
    print("Maximum value found:", -result.fun) # Negate the result to get the actual maximum value

main([1,2,3,4,1,2,3,4],500)