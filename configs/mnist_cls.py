import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import models.model_classifier

# Model part
backbone = {}
backbone["name"] = "models.backbone_mlp.MLP"
backbone["config"] = {
    "input_dim": 784,
    "num_layers": 4,
    "num_hiddens": [512, 256, 128, 64],
    "activation": "relu"
}

backend = {}
backend["name"] = "models.backend_cls.Classifier"
backend["config"] = {
    "input_dim": 64,
    "output_dim": 10,
    "num_layers": 1,
    "num_hiddens": []
}

model_config = {
    "backbone": backbone,
    "backend": backend,
}

#model = torchvision.models.resnet18(pretrained=False)
model = models.model_classifier.CLS_Model(**model_config)
print(f"Total model parameters: {model.get_num_params()}")

# Data part
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=8, 
    shuffle=True, 
    num_workers=4
)
print("Construct train dataset with {} samples".format(len(train_dataset)))

test_dataset = datasets.MNIST('data', train=False, transform=transform)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=2
)
print("Construct test dataset with {} samples".format(len(test_dataset)))

# Loss and training part
learning_rate = 0.001
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 10