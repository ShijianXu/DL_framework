import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from datasets.n2n_dataset import Noise2NoiseDataset

# Model part
backbone = torchvision.models.resnet18(pretrained=False)
backend = torch.nn.Identity()
model_config = {
    "backbone": backbone,
    "backend": backend,
    "created": True
}

# Data part
train_dataset = Noise2NoiseDataset('./data/DIV2K_train_80')
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=8
)
print("Construct train dataset with {} samples".format(len(train_dataset)))

test_dataset = Noise2NoiseDataset('./data/DIV2K_valid_20')
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=4, 
    shuffle=False, 
    num_workers=2
)
print("Construct test dataset with {} samples".format(len(test_dataset)))

# # Loss and training part
# learning_rate = 0.0001
# loss = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), learning_rate)
# #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# num_epochs = 20