import math, numpy as np, os
from matplotlib import pyplot as plt
import PIL
import pickle
from scipy.stats import pearsonr
from TwoParticlesQW import initial_condition_to_corr
from KLdivergence import *
from rgb_basis_trans import *
from QW_object import *
from matplotlib import colors

def print_dist(list, title, x, y, saveto):
    plt.clf()
    plt.figure(figsize = (5, 3.5))
    plt.hist(list, density = True, bins = 50)
    plt.title(title + "\navg = " + str(round(np.mean(list),3)) + " std = " + str(round(np.std(list),3)))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(saveto, dpi=500)
    plt.show()
    plt.cla()
    plt.clf()

def calc_mean_per_bin(hist_data, x_edges, y_edges):
    mean_results = []
    for x, i in zip(x_edges[:-1], range(len(x_edges)-1)):
        if sum(hist_data[i,:])==0:
            mean_results.append(0)
        else:
            mean_results.append(sum(hist_data[i,:] * y_edges[:-1])/sum(hist_data[i,:]))
    return mean_results
    

def evaluate_batch_kl(source_path, save_path, max_energy=3, annotate_int = False, annotate_t = False, max_int = 10, max_t = 5, N_images = 1000):
    file_list = os.listdir(source_path) #load the list of image names in the source path
    try:
        file_list.remove(".DS_Store")
    except ValueError:
        pass
    try:
        file_list.remove("parameters.rtf")
    except ValueError:
        pass

    image_list = [np.asarray(PIL.Image.open(source_path + filename)) for filename in
                  file_list[:N_images]]
    image_list_single_color = [np.array([concat_rgb_sum(image[i, j, :]) for i in range(16) for j in range(16)]).reshape((1, 16, 16)) for image in image_list]  # convert each rgb pixel to a single colored pixel
    QW_object_list = [QW_object(image, Jnm = -np.ones((10, 10)), max_E = max_energy, gamma = max_int, psi0 = [0] * 35 + [1] + [ 0] * 19, prop_time = max_t, annotated_gamma = annotate_int, annotated_time = annotate_t, normalize_like_sum = True) for image in image_list_single_color]
    E_list = np.hstack([obj.E for obj in QW_object_list])
    gamma_list = [obj.gamma for obj in QW_object_list]
    t_list = [obj.prop_time for obj in QW_object_list]
    KL_list = [obj.calc_KL() for obj in QW_object_list]

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
    ax1.hist(KL_list, color = "b")
    ax1.set(xlabel = "KL value", ylabel = "Count")
    ax1.set_title("KL scores\nmean = " + str(round(np.mean(KL_list), 2)) + ", std = " + str(round(np.std(KL_list), 3)))
    ax2.hist(E_list, color = "g")
    ax2.set(xlabel = "E value", ylabel = "Count")
    ax2.set_title("Generated energy values\nmean = " + str(round(np.mean(E_list), 2)) + ", std = " + str(round(np.std(E_list), 2)))
    ax3.hist(gamma_list, color = "r")
    ax3.set(xlabel = "Interaction value", ylabel = "Count")
    ax3.set_title("Generated interactions\nmean = " + str(round(np.mean(gamma_list), 2)) + ", std = " + str(round(np.std(gamma_list), 2)))
    ax4.hist(t_list, color = "orange")
    ax4.set(xlabel = "Time value", ylabel = "Count")
    ax4.set_title("Generated time values\nmean = " + str(round(np.mean(t_list), 2)) + ", std = " + str(round(np.std(t_list), 2)))
    
    time_hist, t_x_edges, t_y_edges = np.histogram2d(t_list, KL_list, bins = 50) #, range = [[0, 2.5], [0, 0.2]])
    t_img_cb = ax5.imshow(time_hist.T, extent = [t_x_edges[0], t_x_edges[-1], t_y_edges[0], t_y_edges[-1]], origin = "lower", cmap = "BuGn", aspect = "auto", label = "KL histogram", norm=colors.LogNorm())
    fig.colorbar(t_img_cb, ax = ax5)
    ax5.set(xlabel = "Time value", ylabel = "KL value")
    ax5.set_title("Time-KL correlation (heatmap)\npearson's correlation = " + str(round(pearsonr(t_list, KL_list)[0], 2)))
    ax5.set(ylim = (0, 1))
    print(calc_mean_per_bin(time_hist, t_x_edges, t_y_edges))

    ax5.plot(t_x_edges[:-1], calc_mean_per_bin(time_hist, t_x_edges, t_y_edges), label = "Mean KL for a given time", color = "red")
    ax5.legend()
    gamma_hist, g_x_edges, g_y_edges = np.histogram2d(gamma_list, KL_list, bins = 50)
    g_img_cb = ax6.imshow(gamma_hist.T, extent = [g_x_edges[0], g_x_edges[-1], g_y_edges[0], g_y_edges[-1]], origin = "lower", cmap = "BuGn", aspect = "auto")
    fig.colorbar(g_img_cb, ax = ax6)
    ax6.set(xlabel = "interaction value", ylabel = "KL value")
    ax6.set_title("interaction-KL correlation (heatmap)\npearson's correlation = " + str(round(pearsonr(gamma_list, KL_list)[0], 2)))

    plt.tight_layout()
    plt.savefig(save_path + "KL_results.png", dpi = 400)
    plt.show()

    if True:
            print_dist(KL_list, "KL list for t1,2 int 0,10", "KL value", "Probability", save_path + "KL_histogram.png")

