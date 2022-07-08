import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.cls = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.cls(x)
        return out