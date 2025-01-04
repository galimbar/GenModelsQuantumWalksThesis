import matplotlib.pyplot as plt
from torchvision.transforms import Normalize as Norm
import numpy as np

from torch.utils.data import Dataset
from Unet import *
from logger import *



class CorrDataset(Dataset):
  def __init__(self, corr_data, transform = None):
    self.corr_data = corr_data
    self.transform = transform

  def __len__(self):
    return len(self.corr_data)

  def __getitem__(self, index):
    corr = torch.from_numpy(self.corr_data[index, :, :, :])
    if self.transform:
      corr = self.transform(corr)
    return corr, 0

class params(object):
    def __init__(self, n_epochs = 500, batch_size = 30, lr = 0.0003, image_size = 16, dataset_path = "", device = "cuda", name = "Diffusion", t_dim = 256, save_path = "", transform = Norm(0.5, 0.5), test_size = 10000):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.device = device
        self.name = name
        self.test_size = test_size
        self.t_dim = t_dim
        self.save_path = save_path
        self.dataset = CorrDataset(np.load(dataset_path).astype(np.float32)[test_size:, None, :, :], transform = transform)
        self.DataLoader = torch.utils.data.DataLoader(dataset = self.dataset,
                                           batch_size = self.batch_size,
                                           shuffle = True)
        self.test_dataset = CorrDataset(np.load(dataset_path).astype(np.float32)[:test_size, None, :, :], transform = transform)
        self.test_DataLoader = torch.utils.data.DataLoader(dataset = self.test_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = True)

def plot_images(images, plot = True, save = True, save_path = ""):
    x = images.cpu().numpy()
    fig, ax = plt.subplots(1, x.shape[0], figsize = (25,2))
    for i in range(x.shape[0]):
        ax[i].imshow(x[i,0,:,:])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    if save:
        plt.savefig(save_path, dpi = 400)
    if plot:
        plt.show()
    fig.clear()
    
