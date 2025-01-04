
import sys, torch, math, os, pickle, numpy as np
from matplotlib import pyplot as plt
import torch
from Diffusion import *
from KL_for_batch_diffusion import *
from torchvision.transforms import Normalize as Norm

def calc_KL_distribution(network_path, save_path, n_samples = 10, im_size = 16):
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = Unet(c_in = 1, c_out = 1, im_size = im_size, device = device).to(device)
    model.load_state_dict(torch.load(network_path))
    model.eval()
    diffusion = Diffusion(image_size = im_size, device = device)
    x_samples = diffusion.sample(model, n_samples).cpu().numpy()
    dist = evaluate_kl(x_samples, title = "KL distribution for Diffusion model", save_path = save_path)
    return (dist.mean, dist.std)

def sort_by_list(list_of_lists, key_list):
    output = []
    for lst in list_of_lists:
        temp = [i for i, j in sorted(zip(lst, key_list), key=lambda x: x[1])]
        output.append(temp)
    return output


def calc_conv_KL(source_path, save_path, network_suffix = "_epochs.pt",  n_samples = 10, plot = True):
    file_list = os.listdir(source_path)
    file_list = [i for i in file_list if i.endswith(network_suffix)]
    KL_means = []
    KL_stds = []
    e_list = []
    for i, filename in enumerate(file_list):
        e = int(filename.split("_")[-2])
        mean, std = calc_KL_distribution(source_path + filename, save_path = save_path + "epoch_" + str(e), n_samples = n_samples)
        KL_means.append(mean)
        KL_stds.append(std)
        e_list.append(e)
    
    KL_means, KL_stds, e_list = sort_by_list([KL_means, KL_stds, e_list], e_list)

    if plot:
        std_up = np.array(KL_means) + np.array(KL_stds)
        std_down = np.array(KL_means) - np.array(KL_stds)
        plt.plot(e_list, KL_means, color = "green", label = "KL mean")
        plt.plot(e_list, std_up, color = "red", linestyle = "dashed", linewidth = 0.5, label = "KL standard deviation")
        plt.plot(e_list, std_down, color = "red", linestyle = "dashed", linewidth = 0.5)
        plt.xlabel("Epoch")
        plt.ylabel("KL")
        plt.title("KL convergence graph")
        plt.legend()
        plt.savefig(save_path + "KL_conv_plot_" + str(max(e_list)) + "_epochs.png", dpi = 400)
        plt.show()
    
    with open(save_path + "KL_div_dict.pkl", "wb") as f:
        pickle.dump({"KL_means" : KL_means, "KL_stds" : KL_stds, "e_list" : e_list} , f)


    return (KL_means, KL_stds, e_list)
