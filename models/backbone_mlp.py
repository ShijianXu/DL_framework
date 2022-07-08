import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
        input_dim,
        num_layers,
        num_hiddens,
        drop_out=0.9,
        activation='relu'
    ):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, num_hiddens[0]))
        if activation == 'relu':
            layers.append(nn.ReLU())
        for i in range(num_layers-1):
            layers.append(nn.Linear(num_hiddens[i], num_hiddens[i+1]))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out