#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
import cmath
import time
import json

# Inner product function of the given state 'psi' and the product state 'phi'.
def Inner_product_function(phi, psi):
    # Perform some operation on the matrices, e.g., element-wise addition
    result = np.abs(np.vdot(phi, psi))
    return result

# Perform a tensor product operation on the rows of the matrix to obtain a vector representation
# of state 'phi' in the computational basis.


def tensor_product_function(matrix):
    num_rows = matrix.shape[0]
    phi = [1]
    for i in range(num_rows):
        phi = np.kron(phi, matrix[i])
    return phi


def sph2cart(theta_1, phi_1, phi_2, theta_2):
    a_1 = cmath.cos(complex(theta_1 / 2, 0))
    a_2 = cmath.cos(complex(theta_2 / 2, 0))
    beta_1 = np.exp(complex(0, phi_1)) * cmath.sin(complex(theta_1 / 2, 0))
    beta_2 = np.exp(complex(0, phi_2)) * cmath.sin(complex(theta_2 / 2, 0))
    return a_1, beta_1, a_2, beta_2


# Function which produces the result of the gp_minimize with given initial state
# param:    psi         List with the given coeffiicients of the given psi
# paramL    N           The discretization space length
def function_max(psi, N):
    N = 2000
    k = np.arange(1, N + 1)
    h = -1 + 2 / (N - 1) * (k - 1)
    phi = np.arccos(h) - np.pi / 2
    theta = np.zeros(N)
    theta[0] = 0

    for j in range(1, N - 1):
        theta[j] = (theta[j - 1] + 3.6 / np.sqrt(N) * 1 / np.sqrt(1 - h[j]**2)) % (2 * np.pi)
    theta[N - 1] = 0

    # # Divide the real part and imaginary part coefficients
    # psi_real_coef = psi[:4]
    # psi_im_coef = psi[4: 2 * 4]
    # psi_state = np.array([])
    # # Given vector psi
    # for i, j in zip(psi_real_coef, psi_im_coef):
    #     psi_state = np.concatenate((psi_state, np.array([complex(i, j)])))
    # magnitude = np.linalg.norm(psi_state)
    # psi_state = psi_state / magnitude
    ################################################

    # Function which solves the problem for two positions in the spherical space
    # The coefficients are a list of integers which points to a location of the
    # phi and theta sample list of the polar and azimouthian cooridnates
    def q2_problem(coefficients):
        # Sample the polar and azimouthian coordinates from the theta and phi value list
        phi_1 = phi[int(coefficients[0])]
        phi_2 = phi[int(coefficients[2])]
        theta_1 = theta[int(coefficients[1])]
        theta_2 = theta[int(coefficients[3])]

        # find the alpha and beta from the polar coordinates
        a_1, beta_1, a_2, beta_2 = sph2cart(theta_1, phi_1, phi_2, theta_2)

        inner_product = (np.conj(a_1 * a_2)) * psi[0] + (np.conj(a_1 * beta_2)) * psi[1]+ (np.conj(beta_1 * a_2)) * psi[2] + (np.conj(beta_1 * beta_2)) * psi[3]

        # Return the negative becauce we want to maximize
        return -np.abs(inner_product)

    space = [Real(0, N - 1, name=f'x{i+1}') for i in range(4)]
    result = gp_minimize(q2_problem,
                         space,
                         acq_func="PI",
                         n_calls=30,
                         n_initial_points=1)
    return -result.fun

    # The best parameters are the indexes which points to the best phi and theta
    # print("Best parameters:")
    # print(f"Ph1:{phi[int(result.x[0])]}")
    # print(f"theta1:{theta[int(result.x[1])]}")
    # print(f"Ph2:{phi[int(result.x[2])]}")
    # print(f"theta2:{theta[int(result.x[3])]}")

    # Negate the result to get the actual maximum value
    # print("Maximum value found:", -result.fun)
    # a_1_list = []
    # a_2_list = []
    # beta_1_list = []
    # beta_2_list = []
    # for point in result.x_iters:
    #     phi_1 = phi[int(point[0])]
    #     phi_2 = phi[int(point[2])]
    #     theta_1 = theta[int(point[1])]
    #     theta_2 = theta[int(point[3])]
    #     a_1, beta_1, a_2, beta_2 = sph2cart(theta_1, phi_1, phi_2, theta_2)
    #     a_1_list.append(a_1)
    #     a_2_list.append(a_2)
    #     beta_1_list.append(beta_1)
    #     beta_2_list.append(beta_2)
    # plt.scatter(a_1_list, beta_1_list)
    # plt.scatter(a_2_list, beta_2_list)
    # plt.xlabel('Parameter x1')
    # plt.ylabel('Parameter x2')
    # plt.title('Sampled Points in Optimization Process (gp_minimize)')
    # plt.show()


def main():
    # Discretize p into 10 intervals between 0 and 1
    discretized_p = np.linspace(0, 1, 10)

    # Create the vector [p, 0, 0, 1-p]
    vector_list = [[p, 0, 0, 1 - p] for p in discretized_p]

    results = []
    for input in vector_list:
        results.append(function_max(input, 2000))
    return results


if __name__ == '__main__':
    start_time1 = time.time()
    results = main()
    file_path = 'output_qe.json'

    # Open the JSON file in write mode
    with open(file_path, 'w') as file:
        # Use json.dump() to write the list to the file
        json.dump(results, file)

    end_time1 = time.time()
    elapsed_time = end_time1 - start_time1
    print(f"Elapsed time: {elapsed_time} seconds")
    file_path1 = 'time_req.json'

    # Open the JSON file in write mode
    with open(file_path1, 'w') as file1:
        # Use json.dump() to write the list to the file
        json.dump([elapsed_time], file1)
