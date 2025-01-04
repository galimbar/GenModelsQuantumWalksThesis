import logging
import torchvision.transforms
from logger import logger
import numpy as np, math
import torch, torch.nn as nn, os
from matplotlib import pyplot as plt
from torch import optim
from Unet import *
from logger import *
from diff_utils import *


class Diffusion():
    def __init__(self, noise_steps=1000, image_size=16, beta_start=0.0001, beta_end=0.02, device = "cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.device = device
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)


    def noise_images(self, image, t): # "t" is a vector that matches the number of images in "image"
        mean = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        std = torch.sqrt(1-self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(image)
        return mean*image + std*epsilon, epsilon

    def choose_random_times(self, N_times):
        return torch.randint(0, self.noise_steps, (N_times,))

    def sample(self, model, n_samples):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n_samples, 1, self.image_size, self.image_size)).to(self.device)
            for step in range(self.noise_steps-1, -1, -1):
                #t = (torch.ones(n_samples) * step).to(self.device)
                #t = t.long()
                pred_noise = model(x, (step*torch.ones(n_samples)).to(self.device))
                alpha = self.alpha[step]*torch.ones(n_samples)[:, None, None, None].to(self.device)
                alpha_hat = self.alpha_hat[step]*torch.ones(n_samples)[:, None, None, None].to(self.device)
                beta = self.beta[step]*torch.ones(n_samples)[:, None, None, None].to(self.device)
                if step>0: #"0" is the last step here
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = x - ((1-alpha) / torch.sqrt(1-alpha_hat)) * pred_noise
                x = (1 / torch.sqrt(alpha)) * x
                x = x + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) +1)/2
        return x

    def sample_with_middle_steps(self, model, n_samples, n_steps_to_show):
        model.eval()
        x_step_dict = {}
        with torch.no_grad():
            x = torch.randn((n_samples, 1, self.image_size, self.image_size)).to(self.device)
            x_step_dict[self.noise_steps] = (x.clamp(-1, 1) +1)/2
            steps_to_save = list(range(0, self.noise_steps, self.noise_steps//(n_steps_to_show)))
            steps_to_save = steps_to_save + list(range(0, steps_to_save[1], steps_to_save[1]//(n_steps_to_show)))
            
            for step in range(self.noise_steps-1, -1, -1):
                #t = (torch.ones(n_samples) * step).to(self.device)
                #t = t.long()
                pred_noise = model(x, (step*torch.ones(n_samples)).to(self.device))
                alpha = self.alpha[step]*torch.ones(n_samples)[:, None, None, None].to(self.device)
                alpha_hat = self.alpha_hat[step]*torch.ones(n_samples)[:, None, None, None].to(self.device)
                beta = self.beta[step]*torch.ones(n_samples)[:, None, None, None].to(self.device)
                if step>0: #"0" is the last step here
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = x - ((1-alpha) / torch.sqrt(1-alpha_hat)) * pred_noise
                x = (1 / torch.sqrt(alpha)) * x
                x = x + torch.sqrt(beta) * noise
                if step in steps_to_save:
                    x_step_dict[step] = (x.clamp(-1, 1) +1)/2

        model.train()
        #x = (x.clamp(-1, 1) +1)/2
        return x_step_dict

    def denoise(self, model, images, t_start):
        model.eval()
        with torch.no_grad():
            x = images.to(self.device)
            n_samples = images.shape[0]
            for step in range(t_start-1, -1, -1):
                #t = (step*torch.ones(images.shape[0])).to(self.device)
                pred_noise = model(x, (step*torch.ones(n_samples)).to(self.device))
                alpha = self.alpha[step]*torch.ones(n_samples)[:, None, None, None].to(self.device)
                alpha_hat = self.alpha_hat[step]*torch.ones(n_samples)[:, None, None, None].to(self.device)
                beta = self.beta[step]*torch.ones(n_samples)[:, None, None, None].to(self.device)
                if step>0: #"0" is the last step here
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = x - ((1-alpha) / torch.sqrt(1-alpha_hat)) * pred_noise
                x = (1 / torch.sqrt(alpha)) * x
                x = x + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) +1)/2
        return x


def plot_conv_graph(loss, test_loss, epochs, save_path, title):
    plt.figure(figsize = (6,5))
    plt.plot(epochs, loss, color = "green", label = "train loss")
    plt.plot(epochs, test_loss, color = "orange", label = "test loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Convergence graph for " + title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()


def calc_total_loss(model,loader, loss_function, diffusion):
    model.eval()
    with torch.no_grad():
        for i, (images, j) in enumerate(loader):
            temp_loss = []
            images = images.to(diffusion.device)
            t = diffusion.choose_random_times(images.shape[0]).to(diffusion.device)
            x_t, noise = diffusion.noise_images(images, t)
            noise_estimated = model(x_t, t)
            loss = loss_function(noise, noise_estimated)
            temp_loss.append(loss.item())
    model.train()
    return np.mean(temp_loss)

def train(params, add_text=""):
    logger()
    loader = params.DataLoader
    test_loader = params.test_DataLoader
    device = params.device
    model = Unet(c_in = 1, c_out = 1, im_size = params.image_size, device = device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = params.lr)
    loss_function = nn.MSELoss()
    diffusion = Diffusion(image_size = params.image_size, device = device)
    loss_list = []
    loss_list.append(calc_total_loss(model, loader, loss_function, diffusion))
    test_loss_list = []
    test_loss_list.append(calc_total_loss(model, test_loader, loss_function, diffusion))
    image_samples = diffusion.sample(model, n_samples=10)
    plot_images(image_samples, save_path=params.save_path + add_text + "_images_epoch_"+ str(0) + ".png")
    for e in range(params.n_epochs):
        logging.debug("entering epoch number: " + str(e))
        for i, (images, j) in enumerate(loader):
            temp_loss = []
            images = images.to(device)
            t = diffusion.choose_random_times(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            noise_estimated = model(x_t, t)
            loss = loss_function(noise, noise_estimated)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_loss.append(loss.item())
        if (e%20==0):
            image_samples = diffusion.sample(model, n_samples=10)
            plot_images(image_samples, plot = True, save_path=params.save_path + add_text + "_images_epoch_"+ str(e+1) + ".png")
            torch.save(model.state_dict(), params.save_path + add_text + "_diff_model_"+ str(e+1) + "_epochs.pt")
            logging.debug("epoch number: " + str(e) + ", MSE=" + str(round(loss_list[-1],2)))
            np.save(params.save_path+ add_text + "_diff_model_"+ str(params.n_epochs) + "_epochs_loss_list.npy", np.array(loss_list))
        loss_list.append(np.mean(temp_loss))
        test_loss_list.append(calc_total_loss(model, test_loader, loss_function, diffusion))
        #logging.debug("epoch number: " + str(e) + ", MSE=" + str(round(loss_list[-1],2)))
        np.save(params.save_path+ add_text + "_diff_model_"+ str(params.n_epochs) + "_epochs_loss_list.npy", np.array(loss_list))
        np.save(params.save_path+ add_text + "_diff_model_"+ str(params.n_epochs) + "_epochs_test_loss_list.npy", np.array(test_loss_list))
        
    plot_conv_graph(loss_list, test_loss_list, list(range(params.n_epochs))+[params.n_epochs], params.save_path + add_text + "_diff_model_"+ str(params.n_epochs) + "_epochs_convergence.png", params.name)
    image_samples = diffusion.sample(model, n_samples=10)
    plot_images(image_samples, plot = True, save_path=params.save_path +add_text + "_final_ims_"+ str(params.n_epochs) + "_epochs.png")
    torch.save(model.state_dict(), params.save_path +add_text + "_diff_model_"+ str(params.n_epochs) + "_epochs.pt")


