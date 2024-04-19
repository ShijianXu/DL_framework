import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from models.model_normalizing_flow_NICE import NICE
from models.module_coupling_layer import AdditiveCouplingLayer, ScalingLayer
from utils import StandardLogistic
from losses import NICELoss

# Model part
IMG_SIZE = (1, 28, 28)
in_dim = IMG_SIZE[0] * IMG_SIZE[1] * IMG_SIZE[2]
num_flows = 4
flow_layers = [AdditiveCouplingLayer(
                    net=nn.Sequential(
                        nn.Linear(in_dim//2, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, in_dim//2)
                    ),
                    mode=(i%2==0)
                ) for i in range(num_flows)]

scale_layer = ScalingLayer(in_dim)

prior = StandardLogistic()

model_config = {
    "in_dim": in_dim,
    "flows": flow_layers,
    "scale_layer": scale_layer,
    "prior": prior
}
model = NICE(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part

transform = transforms.Compose([
    transforms.ToTensor(),
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
num_epochs = 500
learning_rate = 1e-3

loss = NICELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.01), eps=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Sample part, for compatibility with the Trainer
sample_valid = True
sample_valid_freq = 5   # sample images every 5 epochs


# Callbacks part
from callbacks import CheckpointResumeCallback

callbacks = [
    CheckpointResumeCallback(resume=True)
]