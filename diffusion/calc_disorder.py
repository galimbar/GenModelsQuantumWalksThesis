import numpy as np
from Diffusion import *
from Unet import *


def calc_disorder(network_path, save_path, n_samples = 10, im_size = 16):
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = Unet(c_in = 1, c_out = 1, im_size = im_size, device = device).to(device)
    model.load_state_dict(torch.load(network_path))
    model.eval()
    diffusion = Diffusion(image_size = im_size, device = device)
    E_samples = diffusion.sample(model, n_samples).cpu()[:,:, 3:13, 2].numpy().flatten()*3
    plt.hist(E_samples)
    plt.title("Disorder in the generated images")
    plt.xlabel("E")
    plt.ylabel("Count")
    plt.savefig(save_path + "disorder.png", dpi = 400)
    plt.show()
    plt.clf()
    
