import numpy as np
from QW_object import *
from matplotlib import pyplot as plt

class distribution(object):
    def __init__(self, list, title):
        self.list = list
        self.title = title
        self.mean = np.mean(list)
        self.std = np.std(list)

    def print_distribution(self, xlabel, ylabel, save_to, show = False, bins = "auto"):
        plt.hist(self.list, bins = bins)
        plt.title(self.title + "\nmean = " + str(round(self.mean, 3)) + " std = " + str(round(self.std, 3)))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(save_to, dpi=400)
        if show:
            plt.show()
        plt.cla()


def evaluate_kl(images, title, save_path, show = True, Jnm = -np.ones((10, 10)), max_E = 3, max_gamma = 3, gamma = 3, psi0 = [0] * 35 + [1] + [ 0] * 19, prop_time = 2, annotated_gamma = False):
    QW_object_list = [QW_object(images[i, :, :, :], Jnm = Jnm, max_E = max_E,  gamma = gamma, psi0 = psi0, prop_time = prop_time, annotated_gamma = annotated_gamma) for i in range(images.shape[0])]
    KL_list = [QW.calc_KL() for QW in QW_object_list]
    dist = distribution(KL_list, title)
    dist.print_distribution("KL value", "Count", save_path + "KL_distribution.png", show)
    return dist


def evaluate_kl_with_labels(images, time_labels, interaction_labels, title, save_path, show = True, Jnm = -np.ones((10, 10)), max_E = 3, gamma_range = [0,10], time_range = [2,5], psi0 = [0] * 35 + [1] + [ 0] * 19):
    
    if time_range[1] == time_range[0]:
        time_div_range = [0, time_range[1]]
    else:
        time_div_range = time_range
    if gamma_range[1] == gamma_range[0]:
        gamma_div_range = [0, gamma_range[1]]
    else:
        gamma_div_range = gamma_range
    
    QW_object_list = [QW_object(images[i, :, :, :], Jnm = Jnm, max_E = max_E, gamma = (interaction_labels[i]*(gamma_div_range[1]-gamma_div_range[0])) + gamma_div_range[0], psi0 = psi0, prop_time = (time_labels[i]*(time_div_range[1]-time_div_range[0]))+time_div_range[0], annotated_gamma = False, annotated_time = False) for i in range(images.shape[0])]
    KL_list = [QW.calc_KL() for QW in QW_object_list]
    dist = distribution(KL_list, title)
    dist.print_distribution("KL value", "Count", save_path + "KL_distribution.png", show)
    return dist

