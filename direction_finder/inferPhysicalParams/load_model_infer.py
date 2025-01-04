from model import *
from train import *
import torch
import numpy as np


def load_model_infer(saved_model_path, layer_sizes = [110, 128, 64, 32, 16, 8, 4, 2]):
    model = Model(layer_sizes)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    return model

def check_load(data_path, N_samples, model):
    with torch.no_grad():
        corr, E, interaction_list, file_list, time_list = load_images(source_path=data_path, annotated_int=True,
                                                                      constant_int=10, max_energy=3, annotated_t=True,
                                                                      constant_t=2)
        flat_data = np.vstack([np.concatenate([i.flatten(), j], axis=0) for i, j in zip(corr, E)])[:N_samples, :]
        labels = np.vstack([np.array(interaction_list) / 10, np.array(
            time_list) / 2]).T[:N_samples, :] # normalize the interaction and time to have the same scale - up to 1.

        predicted_labels = model(torch.Tensor(flat_data)).numpy()
        for i in range(N_samples):
            print("predicted: " + str(predicted_labels[i, :]) + "   real: " + str(labels[i, :]))


def inferParams(model, images):
    with torch.no_grad():
        corr = [images[i, 3:13, 3:13] for i in range(images.shape[0])]
        E = [images[i, 3:13, 2] for i in range(images.shape[0])]
        flat_data = np.vstack([np.concatenate([i.flatten(), j], axis=0) for i, j in zip(corr, E)])
        predicted_labels = model(torch.Tensor(flat_data)).numpy()
        return predicted_labels

