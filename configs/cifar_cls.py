import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import models.model_classifier

# Model part
backbone = {}
backbone["name"] = "models.backbone_cnn.SimpleCNN"
backbone["config"] = {
    "input_channels": 3,
    "output_channels": 16
}

backend = {}
backend["name"] = "models.backend_cls.Classifier"
backend["config"] = {
    "input_dim": 16*5*5,    # sepcify according to the SimpleCNN structure and input size
    "output_dim": 10,
    "num_layers": 3,
    "num_hiddens": [120, 84]
}

model_config = {
    "backbone": backbone,
    "backend": backend,
}

model = models.model_classifier.CLS_Model(**model_config)
print(f"Total model parameters: {model.get_num_params()}")

# Data part
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=8
)
print("Construct train dataset with {} samples".format(len(train_dataset)))

valid_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=2
)
print("Construct test dataset with {} samples".format(len(valid_dataset)))

test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=2
)
test_require_gt = True

# Loss and training part
learning_rate = 0.001
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = None
num_epochs = 20