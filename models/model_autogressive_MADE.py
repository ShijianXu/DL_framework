# https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/modules/flows.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.weight*self.mask, self.bias)


# https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/modules/flows.py#L10
def create_masks(
    input_size, hidden_size, n_hidden, input_order="sequential", input_degrees=None
):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == "sequential":
        degrees += (
            [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += (
            [torch.arange(input_size) % input_size - 1]
            if input_degrees is None
            else [input_degrees % input_size - 1]
        )

    elif input_order == "random":
        degrees += (
            [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += (
            [torch.randint(min_prev_degree, input_size, (input_size,)) - 1]
            if input_degrees is None
            else [input_degrees - 1]
        )

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MADE(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 n_hidden, 
                 out_features):
        super(MADE, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.n_hidden = n_hidden
        self.out_features = out_features

        # create masks
        masks, self.input_degrees = create_masks(
            in_features, hidden_features, n_hidden
        )

        # construct the model
        self.net_input = MaskedLinear(in_features, hidden_features, masks[0])

        self.net = []
        for m in masks[1:-1]:
            self.net.append(nn.ReLU())
            self.net.append(MaskedLinear(hidden_features, hidden_features, m))

        self.net.append(nn.ReLU())
        self.net.append(MaskedLinear(hidden_features, out_features, masks[-1]))

        self.net = nn.Sequential(*self.net)

    def compute_loss(self, x, output, criterion):
        # binary cross entropy loss
        loss = criterion(output, x)
        return {"loss": loss}

    def process_batch(self, batch, criterion, device):
        """
        Process a batch of images through the normalizing flow.
        Args:
            batch: torch.Tensor, a batch of images
            criterion: nn.Module, the loss function to use
            device: str, the device to use
        """
        x = batch[0].to(device)
        x = x.view(x.size(0), -1)

        output = self.net(self.net_input(x))
        loss = self.compute_loss(x, output, criterion)
        return loss

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def generate(self, num_samples, device):
        samples = torch.zeros(num_samples, self.in_features).to(device)
        # autoregressive model, sample one pixel at a time
        with torch.no_grad():
            for i in range(self.in_features):
                logits = self.net(self.net_input(samples))
                probas = torch.sigmoid(logits)
                pixel_i_samples = torch.bernoulli(probas[:, i])
                samples[:, i] = pixel_i_samples

        return samples.view(-1, 1, 28, 28)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def everything_to(self, device):
        pass

    def reset_metric(self):
        pass


if __name__ == "__main__":
    model = MADE(3, 4, 1, 6)
    
    print(f"Total model parameters: {model.get_num_params()}")