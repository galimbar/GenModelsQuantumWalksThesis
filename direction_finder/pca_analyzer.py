import sys, torch, math, os, pickle, numpy as np, PIL, logging
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, binned_statistic_2d
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
from scipy.ndimage import gaussian_filter

from load_images import *
from KLdivergence import *
from KLscore_for_batch import *
from TwoParticlesQW import *

from load_model_infer import *


logging.basicConfig(level=logging.INFO,
                           format='%(asctime)s %(levelname)s %(message)s',
                           datefmt="%Y-%m-%d %H:%M:%S",
                           filename="ganspace_log.log")

"""
LOAD NETWORK, PERFORM PCA, GENERATE IMAGES:
"""

"""
here i start with code of gradient calculation, some support functions:
"""


def calc_grad(data, PCs, param_values, bins = 10, n_pcs = 2, gaussian_smooth = False):
    w_list, int_list, t_list, disorder_list, lattice_list = data
    principal_components, w_pca, explained_variance, explained_variance_ratio = PCs
    w_projected_on_pcs = np.vstack([np.sum(w_list * i, axis = 1) for i in principal_components]).swapaxes(0,1)
    statistic, bin_edges, binnumber = sp.stats.binned_statistic_dd(w_projected_on_pcs[:, :n_pcs], param_values, statistic="mean", bins = bins)

    bin_scale = [np.mean([i[j+1]-i[j] for j in range(len(i)-1)]) for i in bin_edges]
    if gaussian_smooth:
        statistic = gaussian_filter(statistic, 1)

    grad = np.gradient(statistic, *bin_scale)
    return (grad, bin_edges, bin_scale)

def project_w_on_pc(w, pc):
    w_projected_on_pc = np.sum(w * pc)
    return w_projected_on_pc

def find_location_in_grad_matrix(value, bin_edges):
    for i in range(len(bin_edges)-1):
        if (value>=bin_edges[i] and value<bin_edges[i+1]):
            return i
    return np.NaN

def extract_val(array, index_list):
    temp = array.copy()
    for i in index_list:
        temp = temp[i]
    return temp

def norm_vector(vec):
    return np.array(vec)/(np.sum([i**2 for i in vec])**0.5)

def remove_vector_b_from_a(a, b):
    return norm_vector(a - (np.sum(a*b))*b)
    

def move_on_grad(grad, bin_edges, bin_scale, initial_w, principal_components, sign = 1, disentangle = False, grad_disentanglement = ""):
    pc_projections = [project_w_on_pc(initial_w, pc) for pc in principal_components[:len(bin_scale), :]]
    locations_in_grad_matrix = [find_location_in_grad_matrix(val, edg) for val, edg in zip(pc_projections, bin_edges)]

    """
    here we use the fact that all of N-dimentional surface vectors perpendicular to the gradient, represent directions of zero change, by definition.

    """
    grad_values_in_point = norm_vector([extract_val(pc_grad, locations_in_grad_matrix) for pc_grad in grad])
    print("\npre-disentanglement grad:")
    print(grad_values_in_point)

    if disentangle:
        grad_values_in_point_to_remove = norm_vector([extract_val(pc_grad, locations_in_grad_matrix) for pc_grad in grad_disentanglement])

        print("\ndisentanglement grad to remove:")
        print(grad_values_in_point_to_remove)

        grad_values_in_point = remove_vector_b_from_a(grad_values_in_point, grad_values_in_point_to_remove)    
    
    print("\npost-disentanglement grad:")
    print(grad_values_in_point)
    move_rate = min(bin_scale)
    movement = sum([move_rate*grad_val*pc for grad_val, pc in zip(grad_values_in_point, principal_components[:len(bin_scale)])])
    return initial_w + (sign * movement)

def create_set_of_w_using_grad(data, PCs, value_list, initial_w, bins = 10, n_pcs = 2, save_grad_path = "", disentangle = False, value_to_disentangle = ""):
    grad, bin_edges, bin_scale = calc_grad(data, PCs, value_list, bins, n_pcs)
    
    if disentangle:
        grad_disentanglement, bin_edges_disentanglement, bin_scale_disentanglement = calc_grad(data, PCs, value_to_disentangle, bins, n_pcs)
    
    with open(save_grad_path + "grad.pkl", "wb") as f:
            pickle.dump((grad, bin_edges, bin_scale), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    principal_components, w_pca, explained_variance, explained_variance_ratio = PCs
    
    print("bin scale for all pcs in order: " + str(bin_scale))
    print("move scale for each step: " + str(min(bin_scale)))

    positive_w_list = []
    negative_w_list = []
    
    for initial_w_instance in initial_w:
        temp_pos_w_list = []
        temp_neg_w_list = []

        temp_w = initial_w_instance.copy()
        while True:
            try:
                temp_w = move_on_grad(grad, bin_edges, bin_scale, temp_w, principal_components, 1, disentangle = disentangle, grad_disentanglement = grad_disentanglement)
            except IndexError:
                print("breaking")
                break
            if not np.isnan(temp_w).any():
                temp_pos_w_list.append(temp_w)
                #logging.info("pos list length: " + str(len(positive_w_list)))

                if len(temp_pos_w_list)>10*bins:
                    print("breaking")
                    break
                #logging.info("length of positive w list: " + str(len(positive_w_list)))
            else:
                print("breaking")
                break

        temp_w = initial_w_instance.copy()
        while True:
            try:
                temp_w = move_on_grad(grad, bin_edges, bin_scale, temp_w, principal_components, -1, disentangle = disentangle, grad_disentanglement = grad_disentanglement)
            except IndexError:
                print("breaking")
                break
            if not np.isnan(temp_w).any():
                temp_neg_w_list.append(temp_w)
                #logging.info("neg list length: " + str(len(negative_w_list)))
                if len(temp_neg_w_list)>10*bins:
                    print("breaking")
                    break

            else:
                break
        positive_w_list.append(temp_pos_w_list)
        negative_w_list.append(temp_neg_w_list)

    return (positive_w_list, negative_w_list)

"""
the support code of the gradients ends here
"""



def generate_z_batch(n_samples, G, device = "cuda"):
    z = torch.nn.functional.normalize(torch.randn([n_samples, G.z_dim]), dim=-1).to(device)
    return z

def get_w_from_z(z, G, save_mean_w = True, mean_w_path = ""):
    w = G.mapping(z, c=None)
    if save_mean_w:
        mean = torch.mean(w, dim=0, keepdim=True)
        torch.save(mean, mean_w_path)
        distance = w - mean
        distance = distance*distance
        if len(list(distance.shape))==3:
            distance = distance[:, 0, :]
        distance = torch.sum(distance, dim=1)**0.5
        distance = distance.cpu().numpy()
        plt.show()
        plt.clf()
        plt.hist(distance)
        plt.title("w length statistics\nmean = " + str(round(np.mean(distance),2)) + " and std = " + str(round(np.std(distance),2)))
        plt.xlabel("distance")
        plt.ylabel("count")
        plt.savefig(mean_w_path[:-2] + "png")
        plt.show()
        plt.clf()

    return w

def get_principal_components(w, n_components = "all"):
    w_pca = PCA(n_components = (None if n_components=="all" else n_components))
    w_pca.fit(w)
    principal_components = w_pca.components_
    return (principal_components, w_pca, w_pca.explained_variance_, w_pca.explained_variance_ratio_)

def align_w(w, G, inversemapper):
    return G.mapping(inversemapper(w, 0), c=None)

def move_w_on_pc(initial_w, component, n_steps, total_scale, G, inversemapper = "", device = "cuda"):
    movements = np.arange(0, total_scale/2 + total_scale/n_steps, total_scale/n_steps)
    movements = np.concatenate((np.flip(-movements[1:]), movements))
    edited_w = np.vstack([initial_w[:,0,:] + i * component for i in movements])
    edited_w = np.stack([edited_w for i in range(6)], axis = 1)
    edited_w = torch.from_numpy(edited_w).to(device)

    return edited_w, movements

def synth_images(w, G):
    image_results = []
    n_batches = 1000
    if w.shape[0]<=n_batches:
        n_batches = 1
    res = int(w.shape[0]%n_batches)
    chunk = w.shape[0]//n_batches
    for i in range(0, w.shape[0] - res, chunk):
        if i!=w.shape[0] - res - 1:
            temp_images = G.synthesis(w[i:i+chunk,:,:])
        else:
            temp_images = G.synthesis(w[i:,:,:])
        temp_images = (temp_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        image_results.append(temp_images)
    return np.vstack(image_results)

def save_set_of_images(images, path, n_pc, scale, movements, N_unique_initial_w):
    movement_with_unique_image = np.hstack([np.array([str(j) + "_uniq_im_" + str(i) for i in range(N_unique_initial_w)]) for j in movements])
    for image, i in zip(images, movement_with_unique_image):
        PIL.Image.fromarray(image, "RGB").save(path + ("pc_" + str(n_pc) + "_movement_" + str(i)).replace(".",",") + ".png")

def save_set_of_images_for_grad(images, path, n_pc, scale, movements, current_N_unique_initial_w):
    for image, i in zip(images, movements):
        PIL.Image.fromarray(image, "RGB").save(path + ("pc_" + str(n_pc) + "_movement_" + str(i) + "_uniq_im_" + str(current_N_unique_initial_w)).replace(".",",") + ".png")

def project_to_plane(PCs_to_project, pcs, w_points, save_path):
    pc1_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[0], :], axis = 1)
    pc2_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[1], :], axis = 1)
    hist, g_x_edges, g_y_edges = np.histogram2d(pc1_projection, pc2_projection, bins = 50)
    colorbar = plt.imshow(hist.T, extent = [g_x_edges[0], g_x_edges[-1], g_y_edges[0], g_y_edges[-1]], origin = "lower", aspect = "auto")
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    plt.title("Projection of all W points on PCs: " + str(PCs_to_project)[1:-1])
    plt.savefig(save_path, dpi=400)
    plt.close() 


def find_w_within_range(w, G, images, E_range, t_range, int_range, inferParamsFromImage, inferPhysicalParamsModelPath, save_results = False, savepath = ""):
    divide_by = 255*3
    if inferParamsFromImage:
        model = load_model_infer(inferPhysicalParamsModelPath)
        temp_imag = np.sum(images.copy(), axis = 3)/divide_by
        temp_imag[:, 3:13, 3:13] = 2 * (temp_imag[:, 3:13, 3:13] **2 )
        temp_imag[:, 3:13, 2] = temp_imag[:, 3:13, 2]*3

        estimated_params = inferParams(model, temp_imag)
        int_vals = estimated_params[:, 0]*10
        t_vals = estimated_params[:, 1]*2
        
    else:
        t_vals = np.sum(images[:, 14, 14, :], axis = 1)*2/divide_by
        int_vals = np.sum(images[:, 1, 14, :], axis=1)*10/divide_by
    
    E_vals = np.sum(images[:, 3:13, 2, :], axis=2)*3/divide_by
    
    w_list = []
    int_list = []
    t_list = []
    disorder_list = []
    lattice_list = []
    for i, j, k, l in zip(w, t_vals, int_vals, E_vals):
        if (j>t_range[0] and j<t_range[1] and k>int_range[0] and k<int_range[1] and (l>E_range[0]).all() and (l<E_range[1]).all()):

            w_list.append(i)
            int_list.append(k)
            t_list.append(j)
            disorder_list.append(np.std(l))
            lattice_list.append(l)
    
    return_arr = (np.vstack(w_list), int_list, t_list, disorder_list, lattice_list)
    if save_results:
        with open(savepath + "/w_and_values.pkl", "wb") as f:
            pickle.dump(return_arr, f, protocol=pickle.HIGHEST_PROTOCOL)
    return return_arr

def project_int_to_plane(PCs_to_project, pcs, w_points, interactions, const_E, const_t, save_path):
    pc1_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[0], :], axis = 1)
    pc2_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[1], :], axis = 1)
    colorbar = plt.scatter(pc1_projection, pc2_projection, c = interactions)
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    plt.title("Projection of all W points, interaction values\non PCs: " + str(PCs_to_project)[1:-1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    
    with open(save_path[:-4] + "_raw_data.pkl", "wb") as f:
        pickle.dump([pc1_projection, pc2_projection, interactions], f, protocol=pickle.HIGHEST_PROTOCOL)


    bins = 50
    hist_count, g_x_edges, g_y_edges = np.histogram2d(pc1_projection, pc2_projection, bins = bins)
    hist_sum, _1, _2 = np.histogram2d(pc1_projection, pc2_projection, weights = interactions, bins = bins)
    mean_hist = hist_sum/hist_count
    colorbar = plt.imshow(mean_hist.T, extent = [g_x_edges[0], g_x_edges[-1], g_y_edges[0], g_y_edges[-1]], origin = "lower", aspect = "auto", cmap = "rainbow")
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    plt.title("Projection of all W points, mean interaction values\non PCs: " + str(PCs_to_project)[1:-1])
    plt.tight_layout()
    plt.savefig(save_path[:-4] + "mean.png", dpi=400)
    plt.close()

    hist_std, g_x_edges, g_y_edges, _  = binned_statistic_2d(pc1_projection, pc2_projection, interactions, statistic = "std", bins = bins)
    colorbar = plt.imshow(hist_std.T, extent = [g_x_edges[0], g_x_edges[-1], g_y_edges[0], g_y_edges[-1]], origin = "lower", aspect = "auto", cmap = "brg")
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    plt.title("Projection of all W points, std interaction values\non PCs: " + str(PCs_to_project)[1:-1])
    plt.tight_layout()
    plt.savefig(save_path[:-4] + "std.png", dpi=400)
    plt.close()


def project_t_to_plane(PCs_to_project, pcs, w_points, times, const_E, const_int, save_path):
    pc1_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[0], :], axis = 1)
    pc2_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[1], :], axis = 1)
    colorbar = plt.scatter(pc1_projection, pc2_projection, c = times)
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    plt.title("Projection of all W points, time values\non PCs: " + str(PCs_to_project)[1:-1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()

    with open(save_path[:-4] + "_raw_data.pkl", "wb") as f:
        pickle.dump([pc1_projection, pc2_projection, times], f, protocol=pickle.HIGHEST_PROTOCOL)



    bins = 50
    hist_count, g_x_edges, g_y_edges = np.histogram2d(pc1_projection, pc2_projection, bins = bins)
    hist_sum, _1, _2 = np.histogram2d(pc1_projection, pc2_projection, weights = times, bins = bins)
    mean_hist = hist_sum/hist_count
    colorbar = plt.imshow(mean_hist.T, extent = [g_x_edges[0], g_x_edges[-1], g_y_edges[0], g_y_edges[-1]], origin = "lower", aspect = "auto", cmap = "rainbow")
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    plt.title("Projection of all W points, mean time values\non PCs: " + str(PCs_to_project)[1:-1])
    plt.tight_layout()
    plt.savefig(save_path[:-4] + "mean.png", dpi=400)
    plt.close()

    hist_std, g_x_edges, g_y_edges, _  = binned_statistic_2d(pc1_projection, pc2_projection, times, statistic = "std", bins = bins)
    colorbar = plt.imshow(hist_std.T, extent = [g_x_edges[0], g_x_edges[-1], g_y_edges[0], g_y_edges[-1]], origin = "lower", aspect = "auto", cmap = "brg")
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    plt.title("Projection of all W points, std time values\non PCs: " + str(PCs_to_project)[1:-1])
    plt.tight_layout()
    plt.savefig(save_path[:-4] + "std.png", dpi=400)
    plt.close()

def project_disorder_to_plane(PCs_to_project, pcs, w_points, disorders, lattices, const_E, const_int, save_path):
    pc1_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[0], :], axis = 1)
    pc2_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[1], :], axis = 1)
    
    colorbar = plt.scatter(pc1_projection, pc2_projection, c = disorders)
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    plt.title("Projection of all W points, disorder values\non PCs: " + str(PCs_to_project)[1:-1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    
    #filters = [np.array([1,0,0,0,0,0,0,0,0,1])+1, 8**abs(np.linspace(-2, 2, 10))]
    pihalf_vec = np.linspace(-math.pi/2, math.pi/2, 10)
    filters = [(np.linspace(-2, 2, 10))**2] + [np.cos(i*pihalf_vec) for i in range(1, 11, 1)] + [np.sin(i*pihalf_vec) for i in range(1, 11, 1)]

    bins = 50
    hist_count, g_x_edges, g_y_edges = np.histogram2d(pc1_projection, pc2_projection, bins = bins)    
    hist_sum, _1, _2 = np.histogram2d(pc1_projection, pc2_projection, weights = disorders, bins = bins)
    mean_hist = hist_sum/hist_count
    colorbar = plt.imshow(mean_hist.T, extent = [g_x_edges[0], g_x_edges[-1], g_y_edges[0], g_y_edges[-1]], origin = "lower", aspect = "auto", cmap = "rainbow")
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    plt.title("Projection of all W points, mean disorder values\non PCs: " + str(PCs_to_project)[1:-1])
    plt.tight_layout()
    plt.savefig(save_path[:-4] + "mean.png", dpi=400)
    plt.close()

    flattened_pc1 = (pc1_projection * np.ones((10,1))).T.flatten()
    flattened_pc2 = (pc2_projection * np.ones((10,1))).T.flatten()
    flattened_lattices = np.array(lattices).flatten()


    hist_std, g_x_edges, g_y_edges, _  = binned_statistic_2d(flattened_pc1, flattened_pc2, flattened_lattices, statistic = "std", bins = bins)
    colorbar = plt.imshow(hist_std.T, extent = [g_x_edges[0], g_x_edges[-1], g_y_edges[0], g_y_edges[-1]], origin = "lower", aspect = "auto", cmap = "brg")
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    plt.title("Projection of all W points, std lattice values\non PCs: " + str(PCs_to_project)[1:-1])
    plt.tight_layout()
    plt.savefig(save_path[:-4] + "std.png", dpi=400)
    plt.close()


    hist_mean_flat, g_x_edges, g_y_edges, _  = binned_statistic_2d(flattened_pc1, flattened_pc2, flattened_lattices, statistic = "mean", bins = bins)
    colorbar = plt.imshow(hist_mean_flat.T, extent = [g_x_edges[0], g_x_edges[-1], g_y_edges[0], g_y_edges[-1]], origin = "lower", aspect = "auto", cmap = "ocean")
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    plt.title("Projection of all W points, mean lattice values (flattened)\non PCs: " + str(PCs_to_project)[1:-1])
    plt.tight_layout()
    plt.savefig(save_path[:-4] + "mean_flattened.png", dpi=400)
    plt.close()

    for filtr, i in zip(filters, range(len(filters))):
        filtered_lattices = [np.sum(j * filtr) for j in lattices]
        hist_mean_filt, g_x_edges, g_y_edges, _  = binned_statistic_2d(pc1_projection, pc2_projection, filtered_lattices, statistic = "mean", bins = bins)
        colorbar = plt.imshow(hist_mean_filt.T, extent = [g_x_edges[0], g_x_edges[-1], g_y_edges[0], g_y_edges[-1]], origin = "lower", aspect = "auto", cmap = "rainbow")
        plt.colorbar(colorbar)
        plt.xlabel("PC number: " + str(PCs_to_project[0]))
        plt.ylabel("PC number: " + str(PCs_to_project[1]))
        plt.title("Projection of all W points, filtered lattice values\nfilter: "+str(filtr)+"\non PCs: " + str(PCs_to_project)[1:-1])
        plt.tight_layout()
        plt.savefig(save_path[:-4] + "mean_filter_number_" + str(i) + ".png", dpi=400)
        plt.close() 

def project_grad_to_pc(PCs_to_project, pcs, w_points, w_movements, G, save_path, param_type, inferParamsFromImage, inferPhysicalParamsModelPath, device = "cuda"):
    pc1_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[0], :], axis = 1)
    pc2_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[1], :], axis = 1)
    images = synth_images(torch.from_numpy(np.stack([w_points for i in range(6)], axis = 1)).to(device), G)
    divide_by = 255*3
    
    if inferParamsFromImage:
        model = load_model_infer(inferPhysicalParamsModelPath)
        temp_imag = np.sum(images.copy(), axis = 3)/divide_by
        temp_imag[:, 3:13, 3:13] = 2 * (temp_imag[:, 3:13, 3:13] **2 )
        temp_imag[:, 3:13, 2] = temp_imag[:, 3:13, 2]*3

        estimated_params = inferParams(model, temp_imag)
        int_vals = estimated_params[:, 0]*10
        t_vals = estimated_params[:, 1]*2

    else:
        t_vals = np.sum(images[:, 14, 14, :], axis = 1)*2/divide_by
        int_vals = np.sum(images[:, 1, 14, :], axis=1)*10/divide_by
    
    zero_move_index = w_movements.index(0)
    if param_type == "interaction":
        color_key = int_vals
    elif param_type == "time":
        color_key = t_vals
    else:
        raise IndexError
    

    plt.scatter([pc1_projection[zero_move_index]], [pc2_projection[zero_move_index]], facecolors='none', edgecolors='r', s = 90, zorder = 5, label = "original image mark")
    colorbar = plt.scatter(pc1_projection, pc2_projection, c = color_key, zorder = 7)
    plt.colorbar(colorbar)
    plt.xlabel("PC number: " + str(PCs_to_project[0]))
    plt.ylabel("PC number: " + str(PCs_to_project[1]))
    
    plt.plot(pc1_projection[:zero_move_index+1], pc2_projection[:zero_move_index+1], c="g", linestyle = "dotted", zorder=1)
    plt.plot(pc1_projection[zero_move_index:], pc2_projection[zero_move_index:], c="g", linestyle = "dotted", zorder=1)



    plt.legend()
    plt.title("gradient results over " + param_type + "\non PCs: " + str(PCs_to_project)[1:-1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()


def project_grad_to_pc_for_paper(PCs_to_project, pcs, w_points, w_movements, G, inferParamsFromImage, inferPhysicalParamsModelPath, projection_save_path, device = "cuda"):
    pc1_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[0], :], axis = 1)
    pc2_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[1], :], axis = 1)
    pc3_projection = np.sum(w_points[:, :]*pcs[PCs_to_project[2], :], axis = 1)

    images = synth_images(torch.from_numpy(np.stack([w_points for i in range(6)], axis = 1)).to(device), G)
    divide_by = 255*3

    if inferParamsFromImage:
        model = load_model_infer(inferPhysicalParamsModelPath)
        temp_imag = np.sum(images.copy(), axis = 3)/divide_by
        temp_imag[:, 3:13, 3:13] = 2 * (temp_imag[:, 3:13, 3:13] **2 )
        temp_imag[:, 3:13, 2] = temp_imag[:, 3:13, 2]*3

        estimated_params = inferParams(model, temp_imag)
        int_vals = estimated_params[:, 0]*10
        t_vals = estimated_params[:, 1]*2

    else:
        t_vals = np.sum(images[:, 14, 14, :], axis = 1)*2/divide_by
        int_vals = np.sum(images[:, 1, 14, :], axis=1)*10/divide_by

    with open(projection_save_path + "projection_grad_raw_data_for_paper.pkl", "wb") as f:
        pickle.dump([pc1_projection, pc2_projection, pc3_projection, w_movements, int_vals, t_vals] , f, protocol=pickle.HIGHEST_PROTOCOL)


"""set n_components = "all" for using all pc. else set it to an int"""
def create_images_with_pca(network_path, n_samples, save_path, save_w_path, projection_save_path, n_steps = 10, total_scale = 1, n_components = "all", initial_w = 'random', add_truncation_ims = True, N_unique_initial_w = 1, graham_schmidt = True, shift_towards_mean = 1, device = "cuda", PCs_to_project = ((0,1), (1,2), (0,2)), mix_PCs =[("one-two", (1,2), (0.5, 0.5), (0.5, -0.5))], random_images_to_show_axis = 10, inferParamsFromImage = False, inferPhysicalParamsModelPath = "", disentangle = False):
    logging.info("\n\n\n\n" + str(datetime.now()) + " performing PCA and creating images...")
    
    logging.info(str(datetime.now()) + " loading network")
    with open(network_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)  # torch.nn.Module
    #inversemapper = torch.load(inversemapper_path).cuda()
    logging.info(str(datetime.now()) + " generating z")
    z = generate_z_batch(n_samples, G, device)
    logging.info(str(datetime.now()) + " generating w")
    w = get_w_from_z(z, G, mean_w_path = save_w_path + "mean_w.pt")
    #print(w.shape)
    all_w_space_images = synth_images(w, G)
    mean_w = torch.load(save_w_path + "mean_w.pt")[0,0,:].cpu().numpy()
    w = w[:,0,:].cpu().numpy()
    logging.info(str(datetime.now()) + " performing pca")
    
    principal_components, w_pca, explained_variance, explained_variance_ratio = get_principal_components(w, n_components)
    
    with open(save_w_path + "/PC_values.pkl", "wb") as f:
        pickle.dump( (principal_components, w_pca, explained_variance, explained_variance_ratio), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    w_axis_lists = []

    logging.info(str(datetime.now()) + " projecting constant E and t images to planes")
    
    for i in range(random_images_to_show_axis):
        chosen_E = np.random.uniform(0,3,10)
        chosen_t = np.random.uniform(1,2)
        margin = 10
        w_list, int_list, t_list, disorder_list, lattice_list = find_w_within_range(w, G, all_w_space_images, E_range = (chosen_E-margin, chosen_E+margin), t_range = (chosen_t-margin, chosen_t+margin), int_range = (0,10), inferParamsFromImage = inferParamsFromImage, inferPhysicalParamsModelPath = inferPhysicalParamsModelPath, save_results = True, savepath = projection_save_path)
        
        w_and_param_data = (w_list, int_list, t_list, disorder_list, lattice_list)
        PCs_data = (principal_components, w_pca, explained_variance, explained_variance_ratio)
        
        #initial_w_for_grad = w_list[i+13192: i+13192+N_unique_initial_w]
        initial_w_for_grad = w_list[i+1811: i+1811+N_unique_initial_w]

        #initial_w_for_grad = w_list[i]
        
        candidates = []
        for gh in range(len(t_list)):
            if (t_list[gh]>1.49 and t_list[gh]<1.51 and int_list[gh]>4.9 and int_list[gh]<5.1):
                candidates.append(gh)
        print("candidate indexes (with t in [1.49, 1.51] and int in [4.9, 5.1]) are: ")
        print(candidates)
        
        
        #bins = 30
        #n_pcs = 4
        bins = 30
        n_pcs = 4
        logging.info("creating set of w using grad for interaction")
        positive_w_list_int, negative_w_list_int = create_set_of_w_using_grad(w_and_param_data, PCs_data, int_list, initial_w_for_grad, bins, n_pcs, save_grad_path = projection_save_path + "int_", disentangle = disentangle, value_to_disentangle = t_list)
        logging.info("outputs: pos list len: " + str(len(positive_w_list_int)) + " neg list len: " + str(len(negative_w_list_int)))
        logging.info("creating set of w using grad for time")
        positive_w_list_t, negative_w_list_t = create_set_of_w_using_grad(w_and_param_data, PCs_data, t_list, initial_w_for_grad, bins, n_pcs, save_grad_path = projection_save_path + "t_", disentangle = disentangle, value_to_disentangle = int_list)
        logging.info("synth images and saving, for interaction grad")
        
        for ww in positive_w_list_int:
            ww.reverse()
        for ww in positive_w_list_t:
            ww.reverse()

        int_w_list_grad = [pos +  [init] + neg for pos, init, neg in zip(positive_w_list_int, initial_w_for_grad,  negative_w_list_int)]
        int_w_list_grad_movements = [list(range(len(pos), 0, -1)) +  [0] + list(range(-1, -1-len(neg), -1)) for pos, init, neg in zip(positive_w_list_int, initial_w_for_grad,  negative_w_list_int)]

        logging.info(" lentgh of combined w list int: " + str(len(int_w_list_grad)))
        logging.info(" lentgh of combined w list int (movements): " + str(len(int_w_list_grad_movements)))
        
        for j in range(N_unique_initial_w):
            images = synth_images(torch.from_numpy(np.stack([np.vstack(int_w_list_grad[j]) for i in range(6)], axis = 1)).to(device), G)
            save_set_of_images_for_grad(images, save_path, "grad-int", total_scale, int_w_list_grad_movements[j], j)

        logging.info("synth images and saving, for time grad")
        
        t_w_list_grad = [pos +  [init] + neg for pos, init, neg in zip(positive_w_list_t, initial_w_for_grad,  negative_w_list_t)]
        t_w_list_grad_movements = [list(range(len(pos), 0, -1)) +  [0] + list(range(-1, -1-len(neg), -1)) for pos, init, neg in zip(positive_w_list_t, initial_w_for_grad,  negative_w_list_t)]

        for j in range(N_unique_initial_w):
            images = synth_images(torch.from_numpy(np.stack([np.vstack(t_w_list_grad[j]) for i in range(6)], axis = 1)).to(device), G)
            save_set_of_images_for_grad(images, save_path, "grad-t", total_scale, t_w_list_grad_movements[j], j)


        for pc_pair in PCs_to_project:
            logging.info("creating time and interaction gradient projection for PCs: " + str(pc_pair))
            project_grad_to_pc(pc_pair, principal_components, np.vstack(int_w_list_grad[0]), int_w_list_grad_movements[0],  G,  projection_save_path + "interaction_projection_gradient_pcs_" + str(pc_pair[0]) + "and" + str(pc_pair[1]) + ".png", param_type = "interaction", device = device, inferParamsFromImage = inferParamsFromImage, inferPhysicalParamsModelPath = inferPhysicalParamsModelPath)
            project_grad_to_pc(pc_pair, principal_components, np.vstack(t_w_list_grad[0]), t_w_list_grad_movements[0], G,  projection_save_path + "time_projection_gradient_pcs_" + str(pc_pair[0]) + "and" + str(pc_pair[1]) + ".png", param_type = "time", device = device, inferParamsFromImage = inferParamsFromImage, inferPhysicalParamsModelPath = inferPhysicalParamsModelPath)

        project_grad_to_pc_for_paper((0, 1, 3), principal_components, np.vstack(int_w_list_grad[0]), int_w_list_grad_movements[0],  G, device = device, inferParamsFromImage = inferParamsFromImage, inferPhysicalParamsModelPath = inferPhysicalParamsModelPath, projection_save_path = projection_save_path + "interaction_")
        project_grad_to_pc_for_paper((0, 1, 3), principal_components, np.vstack(t_w_list_grad[0]), t_w_list_grad_movements[0],  G, device = device, inferParamsFromImage = inferParamsFromImage, inferPhysicalParamsModelPath = inferPhysicalParamsModelPath, projection_save_path = projection_save_path + "time_")


        for pc_pair in PCs_to_project:
            project_int_to_plane(pc_pair, principal_components, w_points = w_list, interactions = int_list, const_E = chosen_E, const_t = chosen_t, save_path = projection_save_path + "int_proj_" + str(i) + "_projection_PCs_" + str(pc_pair[0]) + "and" + str(pc_pair[1])  + ".png")
            project_t_to_plane(pc_pair, principal_components, w_points = w_list, times = t_list, const_E = chosen_E, const_int = chosen_t, save_path = projection_save_path + "time_proj_" + str(i) + "_projection_PCs_" + str(pc_pair[0]) + "and" + str(pc_pair[1])  + ".png")
            project_disorder_to_plane(pc_pair, principal_components, w_points = w_list, disorders =  disorder_list, lattices = lattice_list, const_E = chosen_E, const_int = chosen_t, save_path = projection_save_path + "disorder_proj" + str(i) + "_projection_PCs_" + str(pc_pair[0]) + "and" + str(pc_pair[1])  + ".png")

    logging.info(str(datetime.now()) + " projecting PCs to planes")

    for pc_pair in PCs_to_project:
        project_to_plane(pc_pair, principal_components, w, projection_save_path + "projection_PCs_" + str(pc_pair[0]) + "and" + str(pc_pair[1])  + ".png")
    
    if initial_w == 'random':
        initial_w_value = G.mapping(torch.nn.functional.normalize(torch.randn([N_unique_initial_w, G.z_dim]), dim=-1).to(device), c=None).cpu().numpy()
        torch.save(initial_w_value, save_w_path + "/used_w.pt")
        
    else:
        initial_w_value = initial_w
        N_unique_initial_w = initial_w.shape[0]
    
    logging.info("\n\nmean w shape: " + str(mean_w.shape))
    logging.info(mean_w)
    logging.info("\n\ninitial w shape: " + str(initial_w_value.shape))
    logging.info(initial_w_value)
    
    if shift_towards_mean!=1:
        initial_w_value = (initial_w_value * shift_towards_mean) + (mean_w * (1-shift_towards_mean))

    mean_component = initial_w_value[:,0,:]-np.outer(np.ones([1,N_unique_initial_w]), mean_w) 
    mean_length = (np.sum(mean_component*mean_component, axis=-1))**0.5
    mean_component = mean_component / np.outer(mean_length, np.ones([1, mean_w.shape[0]]))
   

    if add_truncation_ims:
        w_for_pc, movements = move_w_on_pc(initial_w_value, mean_component, n_steps = n_steps, total_scale = total_scale,  G = G, device = device)
        print(w_for_pc.shape)
        images = synth_images(w_for_pc, G)
        save_set_of_images(images, save_path, "mean", total_scale, movements, N_unique_initial_w)
    
    print(str(datetime.now()) + " calculating movement on PCs")
    for pc, n_pc in zip(principal_components, range(len(principal_components))):
        if graham_schmidt:
            pc_mod = pc - (np.outer(np.sum(pc * mean_component, axis = -1), np.ones(mean_component.shape[-1])) * mean_component)
            pc_mod = pc_mod / np.outer((np.sum(pc_mod*pc_mod, axis = -1))**0.5, np.ones(pc_mod.shape[-1]))
            print("\n\npc " + str(n_pc) + "\n\ninner product:\n")
            print(np.sum(pc_mod*mean_component ,axis=-1))
            print("\n\nlength squared: \n")
            print(np.sum(pc_mod*pc_mod , axis=-1))
            print("\n\nall values:\n")
            print(pc_mod)
        else:
            pc_mod = pc
    
        w_for_pc, movements = move_w_on_pc(initial_w_value, pc_mod, n_steps = n_steps, total_scale = total_scale,  G = G, device = device)
        images = synth_images(w_for_pc, G)
        save_set_of_images(images, save_path, n_pc, total_scale, movements, N_unique_initial_w)
    
    for mix in mix_PCs:
        mix_name = mix[0]
        pcs_to_mix = mix[1]
        ratios1 = mix[2]
        ratios2 = mix[3]

        pc_mod_1 = ratios1[0]*principal_components[pcs_to_mix[0], :] + ratios1[1]*principal_components[pcs_to_mix[1], :]
        pc_mod_2 = ratios2[0]*principal_components[pcs_to_mix[0], :] + ratios2[1]*principal_components[pcs_to_mix[1], :]

        w_for_pc, movements = move_w_on_pc(initial_w_value, pc_mod_1, n_steps = n_steps, total_scale = total_scale,  G = G, device = device)
        images = synth_images(w_for_pc, G)
        save_set_of_images(images, save_path, mix_name + "-first", total_scale, movements, N_unique_initial_w)
        
        w_for_pc, movements = move_w_on_pc(initial_w_value, pc_mod_2, n_steps = n_steps, total_scale = total_scale,  G = G, device = device)
        images = synth_images(w_for_pc, G)
        save_set_of_images(images, save_path, mix_name + "-second", total_scale, movements, N_unique_initial_w)

    pc_info = pd.DataFrame.from_dict({"explained_variance" : explained_variance, "explained_variance_ratio" : explained_variance_ratio})
    pc_info.to_csv(save_w_path + "pc_variance_info.csv")
        #print(images.shape)

    
"""
ANALYZE CREATED IMAGES
"""

def extract_info_from_filename(filename_list):
    pc_list = []
    movement_list = []
    unique_im_list = []
    for filename in filename_list:
        name = filename[:-4].split("_")
        movement = float(name[3].replace(",", "."))
        pc = name[1]
        unq_img = name[-1]
        pc_list.append(pc)
        movement_list.append(movement)
        unique_im_list.append(unq_img)
    return movement_list, pc_list, unique_im_list

def organize_data(corr, E, interaction_list, file_list, t_list):
    movement_list, pc_list, unique_im_list = extract_info_from_filename(file_list)
    unique_pc = np.unique(pc_list)  # np.unique also sorts the pcs
    pc_info_dict = {pc:{"interaction":[],"E":[], "corr":[], "movement":[], "t": [],  "unq_img":[]} for pc in unique_pc}
    for pc, index in zip(pc_list, range(len(pc_list))):
        pc_info_dict[pc]["interaction"].append(interaction_list[index])
        pc_info_dict[pc]["E"].append(E[index])
        pc_info_dict[pc]["corr"].append(corr[index])
        pc_info_dict[pc]["movement"].append(movement_list[index])
        pc_info_dict[pc]["unq_img"].append(unique_im_list[index])
        pc_info_dict[pc]["t"].append(t_list[index])

    return pc_info_dict

def calc_correlation_per_pc(pc_info_dict):
    movement_interaction_corr_per_pc = {}
    for pc in pc_info_dict.keys():
        pearson_temp_list = []
        for i in np.unique(pc_info_dict[pc]["unq_img"]):
            uniq_int = []
            uniq_mov = []
            for j in range(len(pc_info_dict[pc]["unq_img"])):
                if pc_info_dict[pc]["unq_img"][j]==i:
                    uniq_int.append(pc_info_dict[pc]["interaction"][j])
                    uniq_mov.append(pc_info_dict[pc]["movement"][j])
    
            try:
                pearson_temp_list.append(pearsonr(uniq_int, uniq_mov)[0])
            except ValueError:
                pearson_temp_list.append(0)
        movement_interaction_corr_per_pc[pc] = np.mean(pearson_temp_list)
    return movement_interaction_corr_per_pc


def vis_KL_list(KL_score_list, L1_dict, mean_KL_per_mov, std_KL_per_mov, min_KL_per_mov, max_KL_per_mov, pc, save_path):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(15, 10))
    ax1.hist(KL_score_list)
    ax1.set(xlabel = "asymmetric KL value", ylabel = "Count")
    ax1.set_title("KL values\nmean = " + str(round(np.mean(KL_score_list), 3)) + ", std = " + str(round(np.std(KL_score_list), 3)))
    ax2.scatter(L1_dict["movement"], mean_KL_per_mov, color = "r", label = "mean KL")
    ax2.scatter(L1_dict["movement"], std_KL_per_mov, color = "g", label = "std of KL")
    ax2.scatter(L1_dict["movement"], max_KL_per_mov, color = "orange", label = "max KL")
    ax2.scatter(L1_dict["movement"], min_KL_per_mov, color = "b", label = "min KL")
    ax2.legend()
    ax2.set(xlabel ="movement on PC", ylabel = "KL score")
    ax2.set_title("mean KL score vs movement")
    ax3.scatter(L1_dict["movement"], L1_dict["disorder_std"], color = "g", label = "disorder std")
    ax3.scatter(L1_dict["movement"], L1_dict["disorder_mean"], color = "blue", label = "disorder mean")
    ax3.legend()
    ax3.set(xlabel ="movement on PC")
    ax3.set_title("disorder vs movement")
    ax4.scatter(L1_dict["movement"], L1_dict["E"], color = "orange")
    ax4.set(xlabel = "movement on PC", ylabel = "mean L1")
    ax4.set_title("mean L1 of the energy lattice from original image")

    ax5.scatter(L1_dict["movement"], L1_dict["t_mean"], color="y", label = "shift mean")
    ax5.scatter(L1_dict["movement"], L1_dict["t_std"], color="g", label = "shift std")
    try:
        t_slope = np.polyfit(L1_dict["movement"], L1_dict["t_mean"], 1)[0]
    except SystemError:
        t_slope = 0
    ax5.set(xlabel = "movement on PC", ylabel = "time shift")
    ax5.legend()
    ax5.set_title("time value shift from original image\nslope = " + str(round(t_slope,4)))
    ax6.scatter(L1_dict["movement"], L1_dict["interaction"], color="purple", label = "shift mean")
    ax6.scatter(L1_dict["movement"], L1_dict["interaction_std"], color="g", label = "shift std")
    try:
        int_slope= np.polyfit(L1_dict["movement"], L1_dict["interaction"], 1)[0]
    except SystemError:
        int_slope = 0
    ax6.set(xlabel = "movement on PC", ylabel = "interaction shift")
    ax6.legend()
    ax6.set_title("interaction value shift from original image\nslope = " + str(round(int_slope,4)))
    plt.suptitle("Results for PC number " + str(pc))
    plt.tight_layout()
    interaction_scale = abs(max(L1_dict["interaction"])-min(L1_dict["interaction"]))
    plt.savefig(save_path + str("KL_results_pc_" + str(pc) + "_interactionscale_" + str(round(interaction_scale)) + ".png"), dpi=400)
    

def calc_KL_per_pc(pc_info_dict, pearson_corr_dict, L1_dict, save_path, pearson_threshold = 0.95):
    pc_above_threshold = []
    for pc, pearson in pearson_corr_dict.items():
        if (abs(pearson)>=pearson_threshold or pearson_threshold==0):
            pc_above_threshold.append(pc)
    print("there are ", len(pc_above_threshold), " principal components with absolute value of pearson's correlation above ", pearson_threshold)
    print("calculating validity of the results...")
    for pc in pc_above_threshold:
        KL_list = [calc_KL_score_asymetric(gamma_exact=normalize_like_sum_image(initial_condition_to_corr(N_sites=10, En=E, Jnm=-np.ones((10, 10)), gamma=interaction, psi0=[0] * 35 + [1] + [0] * 19, prop_time=t)), gamma_estimated=2*corr/sum(sum(corr))) for corr, E, interaction, t  in zip(pc_info_dict[pc]["corr"], pc_info_dict[pc]["E"], pc_info_dict[pc]["interaction"], pc_info_dict[pc]["t"])]
        with open(save_path + "KL_list_PC_" + str(pc) + ".pkl", "wb") as file:
            pickle.dump(KL_list, file, protocol=pickle.HIGHEST_PROTOCOL) 
        
        #calc mean KL for each movement over the image ensemble:
        mean_KL_per_mov = []
        std_KL_per_mov = []
        min_KL_per_mov = []
        max_KL_per_mov = []
        for i in L1_dict[pc]["movement"]:
            temp_kl_list = []
            for kl, mov in zip(KL_list, pc_info_dict[pc]["movement"]):
                if mov == i:
                    temp_kl_list.append(kl)
            mean_KL_per_mov.append(np.mean(temp_kl_list))
            std_KL_per_mov.append(np.std(temp_kl_list))
            min_KL_per_mov.append(min(temp_kl_list))
            max_KL_per_mov.append(max(temp_kl_list))
        vis_KL_list(KL_list,  L1_dict[pc], mean_KL_per_mov, std_KL_per_mov, min_KL_per_mov, max_KL_per_mov,  pc, save_path)

def add_data_to_pc_info(pc_info_path, pearson_corr_dict, pc_to_ignore):
    pc_info = pd.read_csv(pc_info_path)
    pc_info["pearson_corr_to_interaction"] =list({key:pearson_corr_dict[key] for key in sorted(subtract_list(pearson_corr_dict.keys(), pc_to_ignore))}.values())
    pc_info.to_csv(pc_info_path)
    return pc_info

def mean_L1(x,y, abs_val = True):
    if abs_val:
        return np.mean(abs(np.array(x)-np.array(y)))
    else:
        return np.mean(np.array(x)-np.array(y))

def clac_L1_for_list(data_dict, key, zero_movement_df, abs_val = True):
    L1 = []
    for i in range(len(data_dict[key])):
        L1.append(mean_L1(data_dict[key][i], zero_movement_df[zero_movement_df["unq_img"]==data_dict["unq_img"][i]][key].iat[0], abs_val))
    return L1

def subtract_list(original, to_remove):
    return [i for i in list(original) if i not in list(to_remove)]

def group_L1_dict(L1_dict, pc_info_dict):
    grpuped_L1_dict = {pc: {"interaction":[], "interaction_std" : [], "t_mean" : [], "t_std" : [], "disorder_std":[], "disorder_mean":[], "E":[], "corr":[], "movement":[]} for pc in sorted(list(pc_info_dict.keys()))}
    for pc in pc_info_dict.keys():
        for movement in np.unique(pc_info_dict[pc]["movement"]):
            grpuped_L1_dict[pc]["movement"].append(movement)
            temp_E = []
            temp_disorder = []
            temp_corr = []
            temp_inter = []
            temp_t = []
            for i in range(len(pc_info_dict[pc]["movement"])):
                if pc_info_dict[pc]["movement"][i]==movement:
                    temp_E.append(L1_dict[pc]["E"][i])
                    temp_disorder.append(pc_info_dict[pc]["E"][i])
                    temp_corr.append(L1_dict[pc]["corr"][i])
                    temp_inter.append(L1_dict[pc]["interaction"][i])
                    temp_t.append(L1_dict[pc]["t"][i])
            grpuped_L1_dict[pc]["interaction"].append(np.mean(temp_inter))
            grpuped_L1_dict[pc]["interaction_std"].append(np.std(temp_inter))
            grpuped_L1_dict[pc]["E"].append(np.mean(temp_E))
            grpuped_L1_dict[pc]["corr"].append(np.mean(temp_corr))
            grpuped_L1_dict[pc]["disorder_std"].append(np.std(np.hstack(temp_disorder)))
            grpuped_L1_dict[pc]["disorder_mean"].append(np.mean(np.hstack(temp_disorder)))
            grpuped_L1_dict[pc]["t_mean"].append(np.mean(temp_t))
            grpuped_L1_dict[pc]["t_std"].append(np.std(temp_t))

    return grpuped_L1_dict

def calc_L1_data_from_info_dict(pc_info_dict):
    L1_dict = {pc: {"interaction":[],"E":[], "corr":[], "movement":[], "t" : []} for pc in sorted(list(pc_info_dict.keys()))}
    for pc in pc_info_dict.keys():
        for key in L1_dict[pc].keys():
            zero_movement_df = pd.DataFrame(pc_info_dict[pc])
            zero_movement_df = zero_movement_df[zero_movement_df["movement"]==0]
            L1_dict[pc][key] = clac_L1_for_list(pc_info_dict[pc], key, zero_movement_df, (key != "interaction") and (key != "t"))
    L1_dict = group_L1_dict(L1_dict, pc_info_dict) #group items in L1 dict and apply mean/std calculations
    return L1_dict

def plot_projected_w(network_path, n_samples=10000):
    with open(network_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    z = generate_z_batch(n_samples, G)
    w = get_w_from_z(z, G, save_mean_w = False)
    w = w[:,0,:].cpu().numpy()
    
def analyze_images(source_path, save_path, pc_info_path,  pearson_threshold = 0.95, pc_to_ignore = ["mean"], inferParamsFromImage = False, inferPhysicalParamsModelPath = ""):
    logging.info("analyzing images...")
    corr, E, interaction_list, file_list, time_list= load_images(source_path, annotated_int = True, constant_int = 10, max_energy = 3, annotated_t = True, constant_t = 2) 
    if inferParamsFromImage:
        model = load_model_infer(inferPhysicalParamsModelPath)
        image_array_for_infer = np.zeros((len(file_list), 16, 16))
        for i in range(len(file_list)):
            image_array_for_infer[i, 3:13, 3:13] = corr[i]
            image_array_for_infer[i, 3:13, 2] = E[i]
        estimated_params = inferParams(model, image_array_for_infer)
        interaction_list = estimated_params[:, 0]*10
        time_list = estimated_params[:, 1]*2
    pc_info_dict = organize_data(corr, E, interaction_list, file_list, time_list)
    L1_dict = calc_L1_data_from_info_dict(pc_info_dict)
    with open(save_path + "pc_info_dict.pkl", "wb") as file:
        pickle.dump(pc_info_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(save_path + "L1_info_dict.pkl", "wb") as file:
        pickle.dump(L1_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    pearson_corr_dict = calc_correlation_per_pc(pc_info_dict)
    pc_info = add_data_to_pc_info(pc_info_path, pearson_corr_dict, pc_to_ignore = pc_to_ignore)
    
    with open(save_path + "pearson_corr_interaction_movement.pkl", "wb") as file:
        pickle.dump(pearson_corr_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    calc_KL_per_pc(pc_info_dict, pearson_corr_dict, L1_dict, save_path, pearson_threshold = pearson_threshold)



