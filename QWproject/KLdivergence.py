import math, numpy as np



def calc_KL_score_symetric(gamma_exact, gamma_estimated, N_particles=2): # h represents the energies of the 1D lattice and gamma represents the correlation matrix. all as np.array
    L = gamma_exact.shape[0]
    epsilon = (L**-2)*(10**-10)
    return 0.5 * sum([(P+epsilon)*math.log((P+epsilon)/(Q+epsilon)) + (Q+epsilon)*math.log((Q+epsilon)/(P+epsilon)) for P, Q in zip(gamma_estimated.flatten()/N_particles, gamma_exact.flatten()/N_particles)])

def calc_KL_score_asymetric(gamma_exact, gamma_estimated, N_particles=2): # h represents the energies of the 1D lattice and gamma represents the correlation matrix. all as np.array
    L = gamma_exact.shape[0]
    epsilon = (L**-2)*(10**-10)
    return sum([(P+epsilon)*math.log((P+epsilon)/(Q+epsilon)) for P, Q in zip(gamma_estimated.flatten()/N_particles, gamma_exact.flatten()/N_particles)])

