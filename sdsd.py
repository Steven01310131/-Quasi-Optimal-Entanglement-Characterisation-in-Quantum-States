import dask
import numpy as np
import cmath

from dask import delayed
import dask
import time
import random
# Define a simple function

def sph2cart(theta_1, phi_1, phi_2, theta_2):
    a_1 = cmath.cos(complex(theta_1 / 2, 0))
    a_2 = cmath.cos(complex(theta_2 / 2, 0))
    beta_1 = np.exp(complex(0, phi_1)) * cmath.sin(complex(theta_1 / 2, 0))
    beta_2 = np.exp(complex(0, phi_2)) * cmath.sin(complex(theta_2 / 2, 0))
    return a_1, beta_1, a_2, beta_2


psi = [1, 2, 3, 4, 1, 2, 3, 4]
N = 2000
k = np.arange(1, N + 1)
h = -1 + 2 / (N - 1) * (k - 1)
phi = np.arccos(h) - np.pi / 2
theta = np.zeros(N)
theta[0] = 0

for j in range(1, N - 1):
    theta[j] = (theta[j - 1] + 3.6 / np.sqrt(N) * 1 / np.sqrt(1 - h[j]**2)) % (2 * np.pi)
theta[N - 1] = 0

# Divide the real part and imaginary part coefficients
psi_real_coef = psi[:4]
psi_im_coef = psi[4: 2 * 4]
psi_state = np.array([])
# Given vector psi
for i, j in zip(psi_real_coef, psi_im_coef):
    psi_state = np.concatenate((psi_state, np.array([complex(i, j)])))
magnitude = np.linalg.norm(psi_state)
psi_state = psi_state / magnitude

# The coefficients are a list of integers which points to a location of the
# phi and theta sample list of the polar and azimouthian cooridnates

@delayed
def q2_problem(coefficients):
    # Sample the polar and azimouthian coordinates from the theta and phi value list
    phi_1 = phi[int(coefficients[0])]
    phi_2 = phi[int(coefficients[2])]
    theta_1 = theta[int(coefficients[1])]
    theta_2 = theta[int(coefficients[3])]

    # find the alpha and beta from the polar coordinates
    a_1, beta_1, a_2, beta_2 = sph2cart(theta_1, phi_1, phi_2, theta_2)

    inner_product = (np.conj(a_1 * a_2)) * psi_state[0] + (np.conj(a_1 * beta_2)) * psi_state[1]+ (np.conj(beta_1 * a_2)) * psi_state[2] + (np.conj(beta_1 * beta_2)) * psi_state[3]

    # Return the negative becauce we want to maximize
    return -np.abs(inner_product)


# Define a simple function
def generate_random_list():
    return [random.randint(0, 1999) for _ in range(4)]

# Generate multiple random lists
num_lists = 100000  # Set the desired number of lists
inputs = [generate_random_list() for _ in range(num_lists)]
# start_time_2 = time.time()


# # List of values for which you want to calculate squares
# start_time = time.time()
# results = [q2_problem(value) for value in inputs]
# end_time = time.time()
# elapsed_time = end_time - start_time

# print(f"Elapsed time: {elapsed_time} seconds")

start_time1 = time.time()
# Use dask.delayed to create delayed objects for each function call
delayed_results = [q2_problem(value) for value in inputs]

# Compute the results using dask.compute
computed_results = dask.compute(*delayed_results)
end_time1 = time.time()
elapsed_time = end_time1 - start_time1

print(f"Elapsed time: {elapsed_time} seconds")
