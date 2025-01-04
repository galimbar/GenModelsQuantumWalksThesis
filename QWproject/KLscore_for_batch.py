import math, numpy as np, os
from matplotlib import pyplot as plt
import PIL
import pickle

from TwoParticlesQW import initial_condition_to_corr
from KLdivergence import *
from rgb_basis_trans import *
from QW_object import *


def print_dist(list, title, x, y, saveto):
    plt.hist(list)
    plt.title(title + "\navg = " + str(round(np.mean(list),3)) + " std = " + str(round(np.std(list),3)))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(saveto, dpi=400)
    plt.show()
    plt.cla()

def evaluate_batch_kl(source_path, save_path, max_energy=3, annotate_int = False, annotate_t = False, max_int = 3, max_t = 2):
    file_list = os.listdir(source_path) #load the list of image names in the source path
    try:
        file_list.remove(".DS_Store")
    except ValueError:
        pass
    try:
        file_list.remove("parameters.rtf")
    except ValueError:
        pass

    if Flase:
        image_list = [np.asarray(PIL.Image.open(source_path + filename))[3:13,2:13,:] for filename in file_list[:]] #convert the images to a list of numpy arrays and truncate the irrelevant pixels
        image_list_single_color = [np.array([rgb_to_256base(image[i,j,:]) for i in range(10) for j in range(11)]).reshape((10,11)) for image in image_list]
        E_corr_pair_list = [{"E" : image[:,0]*max_energy, "correlation" :2* (image[:,1:])**2} for image in image_list_single_color] # separate into 1d array of the energies and the correlation itselfs
        all_E = []
        for i in E_corr_pair_list:
            all_E = all_E + i["E"].tolist()
        plt.hist(all_E)
        plt.savefig(save_path+str("all_E_256.png"))
        plt.show()
        print("all_E_done")

    image_list = [np.asarray(PIL.Image.open(source_path + filename))[3:13,2:13,:] for filename in file_list[:]] #convert the images to a list of numpy arrays and truncate the irrelevant pixels
    image_list_single_color = [np.array([concat_rgb_sum(image[i,j,:]) for i in range(10) for j in range(11)]).reshape((10,11)) for image in image_list] #convert each rgb pixel to a single colored pixel
    E_corr_pair_list = [{"E" : image[:,0]*max_energy, "correlation" :2* (image[:,1:])**2} for image in image_list_single_color] # separate into 1d array of the energies and the correlation itselfs
    
    # in the next line, theres is **0.5)*765)/765)**2  in "normalize like sum image" because the flooring is done then the number is up to 765:

    KL_score_list = [calc_KL_score_asymetric(gamma_exact=normalize_like_sum_image(initial_condition_to_corr(N_sites=10, En=instance["E"], Jnm=-np.ones((10, 10)),gamma=3, psi0=[0] * 35 + [1] + [0] * 19, prop_time=2)),gamma_estimated=instance["correlation"]) for instance in E_corr_pair_list[:]]  # for all images, compare the results to the expected correlation using kl

    Emax_dist = [max(instance["E"]) for instance in E_corr_pair_list]
    Emin_dist = [min(instance["E"]) for instance in E_corr_pair_list]
    all_E = []
    for i in E_corr_pair_list:
        all_E = all_E + i["E"].tolist()
    corrsum_dist = [sum(sum(instance["correlation"])) for instance in E_corr_pair_list]
    print_dist(Emax_dist, "E max value distribution", "E max", "count", save_path+str("Emax.png"))
    print_dist(all_E, "all E", "E", "count", save_path+str("all_E.png"))

    print_dist(Emin_dist, "E min value distribution", "E min", "count", save_path+str("Emin.png"))
    print_dist(corrsum_dist, "Correlation sum distribution", "correlation sum", "count", save_path+str("Corr_sum.png"))

    print ("for base256=" + str(base256) + ", average KL is: " + str(np.mean(KL_score_list)) + ", and the standard deviation is: " + str(np.std(KL_score_list)))
    plt.hist(KL_score_list)
    plt.ylabel("Count")
    plt.xlabel("Asymmetric KL value")
    plt.title("KL values\nmean = " + str(round(np.mean(KL_score_list),3)) + ", std = " + str(round(np.std(KL_score_list),3)))
    plt.savefig(save_path+str("KL_results.png"), dpi=400)
    plt.show()
    plt.cla()

    with open(save_path + '/KL_value_list.pkl', 'wb') as f:
        pickle.dump(KL_score_list, f)


