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
    "backbone_created": True,
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
    batch_size=16, 
    shuffle=True, 
    num_workers=8
)
print("Construct train dataset with {} samples".format(len(train_dataset)))


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

valid_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=2
)
print("Construct test dataset with {} samples".format(len(valid_dataset)))

test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=2
)
test_require_gt = True

# Loss and training part
learning_rate = 0.0001
num_epochs = 20
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                patience=num_epochs/4, factor=0.5, verbose=True)