import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import models.model_classifier

# Model part
backbone = {}
backbone["name"] = "models.backbone_convmixer.ConvMixer"
backbone["config"] = {
    "dim": 256,
    "depth": 16,             # ConvMixer-256/8 (hidden dim / depth)
    "kernel_size": 8,
    "patch_size": 1,        # CIFAR-10 inputs are so small that we initially only used p = 1
    "num_classes": 10,
    "channels": 3,
}
backend = torch.nn.Identity()

model_config = {
    "backbone": backbone,
    "backend": backend,
    "backbone_created": False,
    "backend_created": True
}

model = models.model_classifier.CLS_Model(**model_config)
print(f"Total model parameters: {model.get_num_params()}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=16
)
print("Construct train dataset with {} samples".format(len(train_dataset)))


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

valid_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=8
)
print("Construct test dataset with {} samples".format(len(valid_dataset)))

test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=8
)
test_require_gt = True

# Loss and training part
learning_rate = 0.0001
num_epochs = 200
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# scheduler_name = 'ReduceLROnPlateau'
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                 patience=num_epochs/4, factor=0.5, verbose=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)