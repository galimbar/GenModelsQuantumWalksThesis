import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.attention = nn.MultiheadAttention(1, 1, batch_first = True)
        self.layer_norm1 = nn.LayerNorm([input_size, 1])
        self.layer_norm2= nn.LayerNorm([input_size, 1])
        self.relu = nn.ReLU()


    def forward(self, x):
        x_original = x.view(-1, self.input_size, 1)
        x = self.layer_norm1(x_original)
        attention, w = self.attention(x, x, x)
        attention = attention + x_original
        attention = self.layer_norm2(attention) + attention
        attention = attention.view(-1, self.input_size)
        return self.relu(attention)


class Model(nn.Module):
    def __init__(self, layers_sizes):
        super().__init__()
        # self.input_size = input_size
        # self.output_size = output_size
        # self.intermediate_layers_sizes = intermediate_layers_sizes
        self.layers_sizes = layers_sizes

        #with self attention:
        # self.layers = nn.ModuleList([nn.Sequential(nn.Linear(self.layers_sizes[i], self.layers_sizes[i+1]), nn.ReLU(), SelfAttention(self.layers_sizes[i+1])) for i in range(len(self.layers_sizes)-1)])
        #without self attention:
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(self.layers_sizes[i], self.layers_sizes[i+1]), nn.ReLU()) for i in range(len(self.layers_sizes)-1)])
        # self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x




