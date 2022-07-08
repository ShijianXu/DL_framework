import torch
import torchvision
from torchvision import datasets, transforms

import models.model_classifier

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
    "output_dim": 10
}

model_config = {
    "backbone": backbone,
    "backend": backend,
}

#model = torchvision.models.resnet18(pretrained=False)
model = models.model_classifier.CLS_Model(**model_config)
print("Model init.")


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    
test_dataset = datasets.MNIST('data', train=False, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
print("Dataset init.")