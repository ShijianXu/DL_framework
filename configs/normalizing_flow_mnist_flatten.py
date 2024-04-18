# try to use the normalizing flow for Iris dataset on MNIST dataset
# the idea is to flatten the MNIST images to 784 features
# and input them to the model as if they are the 784 features of the Iris dataset
# but the training failed because of the following error:
# ValueError: Expected value argument (Tensor of shape (128, 784)) to be within the 
# support (IndependentConstraint(Real(), 1)) of the distribution MultivariateNormal

## it seems the learning rate must be very small


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from models.model_normalizing_flow_iris import Iris_NormalizingFlow
from models.module_coupling_layer import IrisCouplingLayer
from losses import IrisLoss


# Model part
class Conditioner(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        
        self.hidden = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output = nn.Linear(hidden_dim, out_dim*2)      # output 2 values for scale and shift
        
    def forward(self, x):
        x = F.leaky_relu(self.input(x))
        for layer in self.hidden:
            x = F.leaky_relu(layer(x))

        x = self.output(x).chunk(2, dim=-1)
        return x
    

class Conditioner2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Conditioner2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        out = out.chunk(2, dim=-1)
        return out

data_dim = 28*28        # 4 features in the iris dataset
num_flows = 5
prior = torch.distributions.multivariate_normal.MultivariateNormal(
    loc=torch.zeros(data_dim).to(device='cuda'),
    covariance_matrix=torch.eye(data_dim).to(device='cuda')
)

flow_layers = []
# for i in range(num_flows):
#     flow_layers += [IrisCouplingLayer(
#                         net=Conditioner2(input_size=data_dim//2,
#                                         output_size=data_dim,
#                                         hidden_size=250),
#                         split=lambda x: x.chunk(2, dim=-1)
#                     )]

for i in range(num_flows):
    flow_layers += [IrisCouplingLayer(
                        net=Conditioner(
                            in_dim=data_dim//2,
                            out_dim=data_dim//2,
                            hidden_dim=500,
                            num_layers=5
                        ),
                        split=lambda x: x.chunk(2, dim=-1)
                    )]

model_config = {
    "flows": flow_layers,
    "prior": prior,
    "is_image": True
}
model = Iris_NormalizingFlow(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part
transform = transforms.Compose([
    # transforms.Resize((14, 14)),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    'data', 
    train=True, 
    download=True, 
    transform=transform
)

# train_dataset, val_dataset = torch.utils.data.random_split(train_set, [50000, 10000])
train_dataloader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)
print("Construct train dataset with {} samples".format(len(train_dataset)))

valid_dataloader = None


test_dataset = datasets.MNIST(
    'data',
    train=False,
    download=True,
    transform=transform
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)
print("Construct test dataset with {} samples".format(len(test_dataset)))


# Loss and training part
num_epochs = 400
learning_rate = 1e-5

loss = IrisLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


# Sample part, visualize the training process
sample_valid = True
sample_valid_freq = 5   # sample images every 5 epochs