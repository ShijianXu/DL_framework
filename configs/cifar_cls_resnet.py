import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import models.model_classifier

# Model part
backbone = torchvision.models.resnet18(pretrained=False)
# change last layer of ResNet
num_final_in = backbone.fc.in_features
NUM_CLASSES = 10
backbone.fc = torch.nn.Linear(num_final_in, NUM_CLASSES)

backend = torch.nn.Identity()
model_config = {
    "backbone": backbone,
    "backend": backend,
    "created": True
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

test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=2
)
print("Construct test dataset with {} samples".format(len(test_dataset)))

# Loss and training part
learning_rate = 0.0001
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 20