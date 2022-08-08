import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        num_layers,
        num_hiddens,
        drop_out=0.9,
        activation='relu'
    ):
        super().__init__()
        
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, num_hiddens[0]))
            layers.append(nn.ReLU())

            for i in range(1, num_layers-1):
                layers.append(nn.Linear(num_hiddens[i-1], num_hiddens[i]))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(num_hiddens[-1], output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 4:
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
        
        out = self.layers(x)
        return out


if __name__ == '__main__':
    model = Classifier(16*5*5, 10, 3,  [120, 84])
    print(model)

    x = torch.rand(8, 16, 5, 5)
    out = model(x)
    print(out.shape)