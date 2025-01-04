import sys

import torch, logging, torchinfo
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_scheduler

from load_images import *
from KLdivergence import *

from model import *

seed = 0
torch.manual_seed(seed)
def mse(a, b):
    mse = np.mean(((a-b)**2).flatten())
    return mse

class CorrDataset(Dataset):
  def __init__(self, flat_data, labels):
    self.flat_data = flat_data
    self.labels = labels


  def __len__(self):
    return len(self.flat_data)

  def __getitem__(self, index):
    corr = torch.from_numpy(self.flat_data[index, :])
    label = torch.from_numpy(self.labels[index, :])
    return corr, label

class inferParams():
    def __init__(self, data_path, train_size, test_size, epochs, batch_size, save_path, lr = 0.003, layer_sizes = [110, 128, 64, 32, 16, 8, 4, 2], device = "cuda"):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.data_path = data_path
        self.train_size = train_size
        self.test_size = test_size
        self.lr = lr
        self.save_path = save_path

    def load_and_prepare(self):
        corr, E, interaction_list, file_list, time_list = load_images(source_path = self.data_path, annotated_int = True, constant_int = 10, max_energy = 3, annotated_t = True, constant_t = 2)
        flat_data = np.vstack([np.concatenate([i.flatten(), j], axis = 0) for i, j in zip(corr, E)])
        labels = np.vstack([np.array(interaction_list)/10, np.array(time_list)/2]).T # normalize the interaction and time to have the same scale - up to 1.
        #if self.device == "mps":
        if True:
            flat_data = flat_data.astype("float32")
            labels = labels.astype("float32")
        train_dataset = CorrDataset(flat_data[:self.train_size, :], labels[:self.train_size, :])
        test_dataset = CorrDataset(flat_data[self.train_size:self.train_size + self.test_size, :], labels[self.train_size:self.train_size + self.test_size, :])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader

    def calc_total_loss(self, model, loader, loss_func):
        model.eval()
        with torch.no_grad():
            temp_loss = []
            temp_time_loss = []
            temp_int_loss = []
            for data, labels in loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                predicted_labels = model(data)
                loss = loss_func(labels.cpu().numpy(), predicted_labels.cpu().numpy())
                temp_loss.append(loss)
                temp_int_loss.append(loss_func(labels[:, 0].cpu().numpy(), predicted_labels[:, 0].cpu().numpy()))
                temp_time_loss.append(loss_func(labels[:, 1].cpu().numpy(), predicted_labels[:, 1].cpu().numpy()))

        model.train()
        return (np.mean(temp_loss), np.mean(temp_int_loss)**0.5, np.mean(temp_time_loss)**0.5)

    def plot_error_distribution(self, model, loader, savepath):
        model.eval()
        with torch.no_grad():
            time_err = []
            int_err = []
            time_values = []
            int_values = []
            for data, labels in loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                predicted_labels = model(data)

                err = abs(labels.cpu().numpy() - predicted_labels.cpu().numpy())
                int_err.append(err[:, 0])
                time_err.append(err[:, 1])
                labels = labels.to("cpu").numpy()
                int_values.append(labels[:, 0])
                time_values.append(labels[:, 1])
        model.train()
        time_err = np.concatenate(time_err)
        int_err = np.concatenate(int_err)
        int_values = np.concatenate(int_values)
        time_values = np.concatenate(time_values)

        plt.hist(int_err, label="interaction errors (abs)", alpha = 0.5, bins = 60)
        plt.hist(time_err, label = "time errors (abs)", alpha = 0.5, bins = 60)
        plt.xlabel("error")
        plt.ylabel("count")
        plt.yscale("log")
        plt.title("time and interaction errors\n(time and interaction values are normalized to 1)")
        plt.legend()
        plt.savefig(savepath + "final_error_dist.png", dpi = 400)
        plt.show()
        plt.clf()

        hist, x_edges, y_edges = np.histogram2d(10 * int_values, 10 * int_err, bins=50)
        int_cb = plt.imshow(hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                              origin="lower", cmap="BuGn", aspect="auto")
        plt.colorbar(int_cb)
        plt.title("interaction errors")
        plt.xlabel("interaction value")
        plt.ylabel("error")
        plt.savefig(savepath + "final_error_dist_int.png", dpi = 400)
        plt.show()
        plt.clf()

        hist, x_edges, y_edges = np.histogram2d(2 * time_values, 2 * time_err, bins=50)
        time_cb = plt.imshow(hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                              origin="lower", cmap="BuGn", aspect="auto")
        plt.colorbar(time_cb)
        plt.title("time errors")
        plt.xlabel("time value")
        plt.ylabel("error")
        plt.savefig(savepath + "final_error_dist_time.png", dpi = 400)
        plt.show()
        plt.clf()
    
    def plot_conv_graph(self, train_loss, test_loss, int_test_rmse, time_test_rmse, epochs, save_path, title):
        plt.figure(figsize=(6, 5))
        plt.plot(epochs, train_loss, color="green", label="train loss")
        plt.plot(epochs, test_loss, color="orange", label="test loss")
        plt.plot(epochs, int_test_rmse, color="red", label="test interaction RMSE\n(norm to 1)")
        plt.plot(epochs, time_test_rmse, color="blue", label="test time RMSE\n(norm to 1)")

        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.title("Convergence graph for " + title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=400)
        plt.show()
        plt.clf()

    def train(self):

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S",
                            filename=self.save_path + "Diff_log.log")
        logging.info("\n\n\n\n\ntrain started")
        logging.info("device: " + self.device)
        logging.info("model structure: " + str(self.layer_sizes))
        logging.info("starting lr: " + str(self.lr))
        logging.info("batch size: " + str(self.batch_size))
        logging.info("total epochs to train: " + str(self.epochs))



        train_loader, test_loader = self.load_and_prepare()
        model = Model(self.layer_sizes).to(self.device)
        model_summary = torchinfo.summary(model)
        logging.info("model summary:\n" + str(model_summary))
        optimizer = torch.optim.AdamW(model.parameters(), lr = self.lr)
        # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.001, total_iters=self.epochs)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = 0.001**(1/30))

        loss_func = nn.MSELoss()
        train_loss_list = []
        test_loss_list = []
        int_test_rmse_list = []
        time_test_rmse_list = []


        for e in range(self.epochs):
            logging.info("epoch " + str(e) + " started")
            logging.info("current lr: " + str(optimizer.param_groups[0]["lr"]))
            temp_train_loss = []
            for data, labels in train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                predicted_labels = model(data)
                loss = loss_func(labels, predicted_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                temp_train_loss.append(loss.item())
            scheduler.step()


            train_loss_list.append(np.mean(temp_train_loss))
            test_loss, int_test_rmse, time_test_rmse = self.calc_total_loss(model, test_loader, mse)
            test_loss_list.append(test_loss)
            int_test_rmse_list.append(int_test_rmse)
            time_test_rmse_list.append(time_test_rmse)
            logging.info("train loss: " + str(train_loss_list[-1]) + "  test loss: " + str(test_loss_list[-1]))
            print("epoch: " + str(e) + ", train loss: " + str(train_loss_list[-1]) + "  test loss: " + str(test_loss_list[-1]))

            # logging.info("train loss: " + str(train_loss_list[-1]))

        self.plot_conv_graph(train_loss_list, test_loss_list, int_test_rmse_list, time_test_rmse_list, range(1, self.epochs+1, 1), self.save_path + "conv_graph.png", "inferring parameters")
        self.plot_error_distribution(model, test_loader, self.save_path)
        torch.save(model.state_dict(), self.save_path + "final_model.pt")


if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    layer_sizes = [110, 128, 64, 32, 16, 8, 4, 2]
    infer_params = inferParams(data_path = data_path, train_size = 97000, test_size = 3000, epochs = 30 , batch_size = 30, save_path = save_path, lr = 0.003, layer_sizes = layer_sizes, device = device)
    infer_params.train()
