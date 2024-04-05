import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from losses import RealNVPLoss
import models.model_normalizing_flow

# Model part
backbone = {}
backbone["name"] = "models.backbone_normalizing_flow.NormalizingFlow"
backbone["config"] = {
    "input_dim": 28*28,     # input dimension (MNIST images are 28x28)
    "hidden_dims": 256,     # hidden dimensions for each layer
    "num_layers": 5,        # number of layers in the model
}

backend = None   # the backend is not needed

model_config = {
    "backbone": backbone,
    "backend": backend,
    "backbone_created": False,
    "img_shape": [1, 28, 28],
}
model = models.model_normalizing_flow.NF_Model(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part
train_dataset = datasets.MNIST(
    'data', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor(),    # only transform to tensor, no normalization
)

print("train dataset min value: ", train_dataset.data.min())
print("train dataset max value: ", train_dataset.data.max())

train_dataloader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=1
)

print("Construct train dataset with {} samples".format(len(train_dataset)))

# Loss and training part
num_epochs = 100
learning_rate = 1e-4

loss = RealNVPLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)