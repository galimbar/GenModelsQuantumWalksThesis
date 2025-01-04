import math, numpy as np, scipy as sp
from matplotlib import pyplot as plt
from scipy import linalg


# create the state list of N particle, each state is a np array with size N_sites
# for example the state with 2 particles in the 1st place is (2,0,0,0...)
def create_state_list(N_sites):
    state_list=[]
    for i in range(N_sites): # run on all the sites in the lattice
        two_state_temp = np.zeros(N_sites)
        two_state_temp[i] = 2 # first, create a state with 2 particles in th i'th site
        state_list.append(two_state_temp)
        if i!=N_sites-1: #then add states with 1 particle in the i'th site and another in each of the sites after it
            for j in range(i+1, N_sites):
                temp_state = np.zeros(N_sites)
                temp_state[i] = 1
                temp_state[j] = 1
                state_list.append(temp_state)
    return state_list

# ladder operators:
# each state is a tuple of (state, coefficient), for example ((1,0,1,0,0,...), 0.5).
# the state describes directly how many particles are in each site and NOT the states of the hamiltonian matrix
# n is between [1,N] and NOT between [0,N-1]

def a_n(previous_result, n):
    state, previous_coefficient = previous_result
    coefficient = state[n-1]**0.5 #the recieved coefficient is sqrt of the particle number before subtraction
    new_state = np.copy(state) # copy is important. otherwise its just a pointer, and that will ruin the results
    if new_state[n - 1]!=0: # if it its zero, the coefficient will be 0 zero as well.
        new_state[n - 1] -= 1
    return (new_state, coefficient * previous_coefficient)

def a_n_dagger(previous_result, n):
    state, previous_coefficient = previous_result
    coefficient = (state[n-1]+1)**0.5 #the recieved coefficient is sqrt of the particle number after addition
    new_state = np.copy(state) # copy is important. otherwise its just a pointer, and that will ruin the results
    new_state[n - 1] += 1
    return (new_state, coefficient * previous_coefficient)

# E_ij function finds the value of the ij slot in the haniltonian H.
# it receives:
#   - the states (with how many particles are in each site)
#   - the 1D array of the site energies En
#   - the 2D array of site-to-site hopping matrix (but uses just the neighboring-sites values, hard coded)
#   - the in-site interaction intensity gamma
#
#  returns a scalar value E

def E_ij(state1, state2, En, Jnm, gamma):

    E = 0

    for i in range(1, len(state1) + 1):
        state_2_mod, coef = a_n_dagger(a_n((state2,1),i),i) # find "coef" that represents the particle number in the i'th site.
        if np.array_equal(state_2_mod, state1): #should keep on-diagonal values only
            E += En[i-1] * coef # on-site energies
            E += 0.5 * gamma * coef * (coef - 1) # interaction energies

    for i in range(1, len(state1) + 1):
        for j in [i-1, i+1]:
            if j not in [0, len(state1) + 1]: # if j is not on the edges
                state_2_mod, coef = a_n_dagger(a_n((state2,1), j), i)
                if np.array_equal(state_2_mod, state1):
                    E -= Jnm[i-1, j-1] * coef

    return E

#  "create_H" function receives the state list from "create_state_list" function and:
#   - the 1D array of the site energies En
#   - the 2D array of site-to-site hopping matrix (but uses just the neighboring-sites values, hard coded)
#   - the in-site interaction intensity gamma
#
# returns the np array of H

def create_H(state_list, En, Jnm, gamma):
    H = np.zeros((len(state_list), len(state_list)))
    for i in range(len(state_list)):
        for j in range(len(state_list)):
            H[i][j] = E_ij(state_list[i], state_list[j], En, Jnm, gamma)
    return H

# "propagate_in_time" creates the propogator for a certain time t and initial condition psi0.
def propagate_in_time(H, t, psi_0):
    U = sp.linalg.expm(-1j * t * H)
    psi_t = np.matmul(U, psi_0)
    return psi_t

# calc_gamma_qr calculates the value of a specific slot in the correlation matrix gamma.
# receives the final state (in the basis of H, and NOT in how many particles in each site) and the q and r locations
# i.e, "state" is the magnitude of each state and "State list" is the list of the states
# q and r are between [1, N] and NOT between [0,N-1]
# "state" and "state list" must have the same size

def calc_gamma_qr(state, q, r, state_list):
    gamma = 0
    for i, magnitude in zip(range(len(state)), state):
        state_left = np.copy(state_list[i])
        state_right = np.copy(state_left)

        state_right, coef1 = a_n(a_n((state_right,1),r),q)
        state_right, coef2 = a_n_dagger(a_n_dagger((state_right,coef1),r),q)
        gamma += coef2 * (abs(magnitude)**2)
    return gamma

# calculate the whole correlation matrix
def calc_gamma_mat(state, state_list):
    N = len(state_list[0])
    gamma = np.zeros((N,N))
    for q in range(N):
        for r in range(N):
            gamma[q,r] = calc_gamma_qr(state, q+1, r+1, state_list)
    return gamma

# plot the results of gamma
def plot_gamma(gamma, path, title):
    print(" the sum of all gamma matrix elements is: " + sum(sum(gamma)))
    plt.imshow(gamma)
    plt.title(title)
    plt.savefig(path, dpi=350)
    plt.xlabel("position r")
    plt.ylabel("position q")
    plt.show()

# diagonalize H and plot the eigenvectors and eigenenergies
def diagonalize_and_plot(H, pathvec, pathE, title):
    En, Vn = np.linalg.eig(H)
    N = len(En)
    fig, axs = plt.subplots(math.ceil(N/2), 2, figsize=(9,N))
    for i, j in zip(range(N), np.argsort(En)):
        axs[math.floor(i/2),i%2].plot(np.arange(N)+1, Vn[:,j])
        axs[math.floor(i/2),i%2].set_title("E = " + str(round(En[j],2)))
        axs[math.floor(i/2),i%2].set_xlabel("site")
        axs[math.floor(i/2),i%2].set_ylabel("real")
    plt.suptitle(title + "  eigen vectors aand energies")
    plt.savefig(pathvec, dpi=350)
    # plt.tight_layout()
    plt.show()
    plt.plot(np.sort(En), np.arange(N)+1)
    plt.title (title + " - all the energies")
    plt.ylabel("E")
    plt.savefig(pathE, dpi=350)
    plt.show()

# given an initial condition psi_0, and total number of steps "step_num", and the hamiltonian h, plot the propagation of the state in time
# the states do not represent location (there are more states than the number of lattice sites) so it's a hard visualization to understand
def print_prop(h, t_max, psi_0, step_num, path):

    prop_arr = np.vstack([propagate_in_time(h, t, psi_0) for t in np.arange(0,t_max, t_max/step_num)])
    plt.imshow(np.absolute(prop_arr))
    plt.xlabel("position")
    plt.ylabel("time")
    plt.savefig(path, dpi=350)
    plt.show()

# "initial_condition_to_corr" is the main function that combines all of the above:
# N_sites - number of sites in the lattice
# En - the energies on the lattice sites (np array of size N_sites)
# Jnm - the site-to-site hopping matrix (np array of size (N_sites,N_sites))
# gamma - the on-site interaction strength (a scalar)
# psi0 - the initial condition (np array. the size is the number of different states in H. for example, for N_sites=10, the size of psi0 is 55)
# prop_time it the propafation time (a scalar)
# plot result - a boolean.
# plot path - the path to save the image if we chose to plot. (a string including the image name and file type)s
# return_correlation - if true(by default), returns the particle correlation. if false, returns the propogator.
# pre_calc_H = bool, False by default. if True, H components are pre-inserted to the function. H = H_OnSiteEnergies + H_hopping + H_interaction
# H_components = [H_OnSiteEnergies, H_hopping, H_interaction] or any other components, since the function only sums these values so there is no specific number of components


def initial_condition_to_corr(N_sites, En, Jnm, gamma, psi0, prop_time, plot_result = False, plot_path="", return_correlation=True, pre_calc_H = False, H_components = []):
    state_list = create_state_list(N_sites) #create the state list
    if not pre_calc_H:
        H = create_H(state_list, En, Jnm, gamma) #create the hamiltonian H
    else:
        H = sum(H_components)
    if not return_correlation: #if we want to return the correlation:
        return sp.linalg.expm(-1j * prop_time * H)
    correlation_matrix = calc_gamma_mat(propagate_in_time(H, prop_time, psi0), state_list) # calculate the correlation matrix
    if plot_result:
        plot_gamma(correlation_matrix, plot_path,
                   "correlation example")
    return correlation_matrix



