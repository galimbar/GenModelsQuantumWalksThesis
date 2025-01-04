import numpy as np
from KLdivergence import *
from TwoParticlesQW import initial_condition_to_corr

def normalize_like_sum_image(array): # this mimics the effect of flooring the numbers using uint8 for the original array so the comparison would be exact
    return ((np.floor(((array/2)**0.5) * 765)/765)**2)*2

class QW_object(object):
    def __init__(self, image, Jnm = -np.ones((10, 10)), max_E = 3,  gamma = 3, psi0 = [0] * 35 + [1] + [ 0] * 19, prop_time = 2, annotated_gamma = False, annotated_time = False, normalize_like_sum = False):
        self.image = image
        self.Jnm = Jnm
        self.psi0 = psi0
        self.annotated_gamma = annotated_gamma
        self.annotated_time = annotated_time

        if self.annotated_gamma:
            self.gamma = image[0,1, 14] * gamma
        else:
            self.gamma = gamma

        if self.annotated_time:
            self.prop_time = image[0,14, 14] * prop_time
        else:
            self.prop_time = prop_time
        
        self.E = image[0, 3:13, 2] * max_E
        self.corr = np.array(2*(image[0, 3:13, 3:13]**2))
        self.corr = 2 * (self.corr/np.sum(self.corr))
        self.normalize_like_sum = normalize_like_sum
        self.KL = None


    def calc_KL(self):
        if self.normalize_like_sum:
            self.KL = calc_KL_score_asymetric(gamma_exact=normalize_like_sum_image(initial_condition_to_corr(N_sites=10, En=self.E, Jnm=self.Jnm, gamma=self.gamma,psi0=self.psi0, prop_time=self.prop_time)), gamma_estimated=self.corr)
        else:
            self.KL = calc_KL_score_asymetric(gamma_exact=initial_condition_to_corr(N_sites=10, En=self.E, Jnm=self.Jnm, gamma=self.gamma,psi0=self.psi0, prop_time=self.prop_time), gamma_estimated=self.corr)
        return self.KL


