import pickle
import sys
from Diffusion import *
from KL_for_batch_diffusion import *
from diff_utils import *
from matplotlib import pyplot as plt
from Unet import *
from torchvision.transforms import Normalize as Norm
from pylab import imshow, cm, colorbar

def interpolate_once(images, amount, t, model, device = "cuda"):
    diffusion = Diffusion(image_size = 16, device = device)
    noisy_images = diffusion.noise_images(images.to(device), (t * torch.ones(2,)).long().to(device))[0]
    interpolated = ((1-amount) * noisy_images[0, :, :, :] + amount * noisy_images[1, :, :, :])[None, :, :, :]
    clean_image = diffusion.denoise(model, interpolated, t)

    return clean_image

def plot_interpolation(results, save_path, t_steps, lambda_steps):
    t_vals = [int(i) for i in np.linspace(0, 999, t_steps)]
    lambda_vals = np.linspace(0, 1, lambda_steps).tolist()
    fig, ax = plt.subplots(nrows=t_steps, ncols=lambda_steps, sharex=True, sharey=True) 
    t_tick_locs = np.linspace(0,t_steps-1, min(3, t_steps), dtype = int)
    lamb_tick_locs = np.linspace(0,lambda_steps-1, min(3, lambda_steps), dtype = int)
    for lambd, t, image in zip(results["lambda"], results["t"], results["image"]):
        ax[t_vals.index(t), lambda_vals.index(lambd)].imshow(image[0, 0, :, :])
        ax[t_vals.index(t), lambda_vals.index(lambd)].set_xticks([])
        ax[t_vals.index(t), lambda_vals.index(lambd)].set_yticks([])
    for i, t in enumerate(t_vals):
        if i in t_tick_locs:
            ax[i, 0].set_ylabel("t = " + str(t))
    for j, lambd in enumerate(lambda_vals):
        if j in lamb_tick_locs:
            ax[-1, j].set_xlabel("lambda = " + str(round(lambd, 2)))
    plt.suptitle("interpolation results")
    #plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=-0.7, hspace=0)
    #plt.xlabel("lambda")
    #plt.ylabel("t")
    plt.savefig(save_path, dpi = 400)
    plt.show()
    fig.clear()
    plt.clf()

def plot_kl(results, save_path, t_steps, lambda_steps):
    KL_map = np.zeros((t_steps, lambda_steps))
    t_vals = [int(i) for i in np.linspace(0, 999, t_steps)]
    t_tick_locs = np.linspace(0,t_steps-1, min(7, t_steps), dtype = int)
    lamb_tick_locs = np.linspace(0,lambda_steps-1, min(7, lambda_steps), dtype = int)
    lambda_vals = np.linspace(0, 1, lambda_steps).tolist()
    for lambd, t, kl in zip(results["lambda"], results["t"], results["KL"]):
        KL_map[t_vals.index(t), lambda_vals.index(lambd)] = kl
    im = imshow(KL_map, cmap=cm.RdBu)
    colorbar(im)
    plt.title("KL heatmap")
    f = lambda x: round(x, 2)
    f = np.vectorize(f)
    plt.yticks(np.arange(t_steps)[t_tick_locs], np.array(t_vals)[t_tick_locs])
    plt.xticks(np.arange(lambda_steps)[lamb_tick_locs], f(np.array(lambda_vals))[lamb_tick_locs], rotation = 90)
    plt.xlabel("lambda")
    plt.ylabel("t")
    plt.tight_layout()
    plt.savefig(save_path, dpi = 400)
    plt.show()
    plt.clf()
    

def plot_similarity(results, save_path, t_steps, lambda_steps, sim_image, other_image, image_name):
    sim_map = np.zeros((t_steps, lambda_steps))
    t_vals = [int(i) for i in np.linspace(0, 999, t_steps)]
    lambda_vals = np.linspace(0, 1, lambda_steps).tolist()
    t_tick_locs = np.linspace(0,t_steps-1, min(7, t_steps), dtype = int)
    lamb_tick_locs = np.linspace(0,lambda_steps-1, min(7, lambda_steps), dtype = int)
    sim_E = sim_image[3:13, 2]
    similarity_scale = np.mean(abs(other_image[3:13, 2].cpu().numpy()-sim_E.cpu().numpy())) # we'll normalize the similarity with that of the two images
    for lambd, t, im in zip(results["lambda"], results["t"], results["image"]):
        sim_map[t_vals.index(t), lambda_vals.index(lambd)] = np.mean(abs(im[0,0,3:13, 2]-sim_E.cpu().numpy()))/similarity_scale 
        
    
    im = imshow(sim_map, cmap=cm.RdBu)
    colorbar(im)
    f = lambda x: round(x, 2)
    f = np.vectorize(f)
    plt.title("L1 similarity of energy heatmap (for " + image_name + " image)\n(normalized by the L1 of the two original images)")
    plt.yticks(np.arange(t_steps)[t_tick_locs], np.array(t_vals)[t_tick_locs])
    plt.xticks(np.arange(lambda_steps)[lamb_tick_locs], f(np.array(lambda_vals))[lamb_tick_locs], rotation = 90)
    plt.xlabel("lambda")
    plt.ylabel("t")
    plt.tight_layout()
    plt.savefig(save_path, dpi = 400)
    plt.show()
    plt.clf()
    return sim_map

def interpolate_images(images, model, t_steps, lambda_steps, save_path, device = "cuda"): #"images" are normalized in [-1, 1]
    results = {"lambda" : [], "t": [], "image": [], "KL": []}
    normed_images = Norm(0.5, 0.5)(images)
    for amount in np.linspace(0, 1, lambda_steps):
        for t in [int(i) for i in np.linspace(0, 999, t_steps)]:
            x = interpolate_once(normed_images, amount, t, model, device).cpu().numpy() # x is returned in the range [0,1]
            results["lambda"].append(amount)
            results["t"].append(t)
            results["image"].append(x)

    results["KL"] = evaluate_kl(np.concatenate(results["image"]), "KL of interpolation images", save_path).list
    with open(save_path + "interpolation_images.pkl", "wb") as f:
        pickle.dump(results, f)
    plot_interpolation(results, save_path+ "interpolation.png", t_steps, lambda_steps)
    plot_kl(results, save_path+ "interpolation_kl.png", t_steps, lambda_steps)
    sim_left = plot_similarity(results, save_path + "left_similarity.png", t_steps, lambda_steps, sim_image = images[0, 0, :, :], other_image = images[1, 0, :, :], image_name = "left")    
    sim_right = plot_similarity(results, save_path + "right_similarity.png", t_steps, lambda_steps, sim_image = images[1, 0, :, :], other_image = images[0, 0, :, :],  image_name = "right")
    return (results, sim_left, sim_right)

def interpolate_n_times(images, model, t_steps, lambda_steps, save_path, t_to_graph = [0,1,2, 3], device = "cuda"): #"images" are normalized in [-1, 1]
    n_times = int(images.shape[0]//2)
    results_list = []
    sim_left_list = []
    sim_right_list = []
    for i in range(n_times):
        results, sim_left, sim_right = interpolate_images(images[[2*i, 2*i+1], :, :, :], model, t_steps, lambda_steps, save_path + "_im" + str(i) + "_", device)
        results_list.append(results)
        sim_left_list.append(sim_left)
        sim_right_list.append(sim_right)
    
    mean_results = {"lambda" : results["lambda"], "t": results["t"]}
    list_of_kl_lists = [i["KL"] for i in results_list]
    mean_results["KL"] = [np.mean([list_of_kl_lists[j][i] for j in range(len(list_of_kl_lists))]) for i in range(len(list_of_kl_lists[0]))] 
    plot_kl(mean_results, save_path + "_mean_interpolation_kl.png", t_steps, lambda_steps)
    for name, data in zip(["left", "right"], [sum(sim_left_list)/len(sim_left_list), sum(sim_right_list)/len(sim_right_list)]):
        im = imshow(data, cmap=cm.RdBu)
        colorbar(im)
        f = lambda x: round(x, 2)
        f = np.vectorize(f)
        t_vals = [int(i) for i in np.linspace(0, 999, t_steps)]
        lambda_vals = np.linspace(0, 1, lambda_steps).tolist()
        t_tick_locs = np.linspace(0,t_steps-1, min(7, t_steps), dtype = int)
        lamb_tick_locs = np.linspace(0,lambda_steps-1, min(7, lambda_steps), dtype = int)
        plt.title("mean L1 similarity of energy heatmap (for " + name + " image)\n(normalized by the L1 of the two original images)")
        plt.yticks(np.arange(t_steps)[t_tick_locs], np.array(t_vals)[t_tick_locs])
        plt.xticks(np.arange(lambda_steps)[lamb_tick_locs], f(np.array(lambda_vals))[lamb_tick_locs], rotation = 90)
        plt.xlabel("lambda")
        plt.ylabel("t")
        plt.tight_layout()
        plt.savefig(save_path + name + "_mean_similarity.png", dpi = 400)
        plt.show()
        plt.clf()
    
    fig, ((a0,a1), (a2, a3)) = plt.subplots(2,2, figsize = (10,10))
    for name, data in zip(["left", "right"], [sum(sim_left_list)/len(sim_left_list), sum(sim_right_list)/len(sim_right_list)]):
        a0.plot(lambda_vals, data[t_to_graph[0], :], label = "normalized L1\nfrom " + name + " image")
        a1.plot(lambda_vals, data[t_to_graph[1], :], label = "normalized L1\nfrom " + name + " image")
        a2.plot(lambda_vals, data[t_to_graph[2], :], label = "normalized L1\nfrom " + name + " image")
        a3.plot(lambda_vals, data[t_to_graph[3], :], label = "normalized L1\nfrom " + name + " image")

    a0.set_xlabel("Lambda")
    a0.set_ylabel("Normalized L1")
    a0.set_title("t = " + str(t_vals[t_to_graph[0]]))
    a1.set_xlabel("Lambda")
    a1.set_ylabel("Normalized L1")
    a1.set_title("t = " + str(t_vals[t_to_graph[1]]))
    a2.set_xlabel("Lambda")
    a2.set_ylabel("Normalized L1")        
    a2.set_title("t = " + str(t_vals[t_to_graph[2]]))
    a3.set_xlabel("Lambda")
    a3.set_ylabel("Normalized L1")        
    a3.set_title("t = " + str(t_vals[t_to_graph[3]]))
    a0.legend()
    a1.legend()
    a2.legend()
    a3.legend()
    plt.suptitle("Normalized L1 for interpolation using various t values")
    plt.tight_layout()
    plt.savefig(save_path + "interp_lambdas.png", dpi = 400)
    plt.show()
    plt.clf()

def load_model(path):
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = Unet(c_in=1, c_out=1, im_size=16, device=device).to(device)
    model.load_state_dict(torch.load(network_path))
    model.eval()
    return model


