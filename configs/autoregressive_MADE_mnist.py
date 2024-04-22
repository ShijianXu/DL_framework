import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from models.model_autogressive_MADE import MADE

# Model part
IMG_SIZE = (1, 28, 28)
in_dim = IMG_SIZE[0] * IMG_SIZE[1] * IMG_SIZE[2]
num_flows = 4


model_config = {
    "in_features": in_dim,
    "hidden_features": 512,
    "n_hidden": 2,
    "out_features": in_dim
}
model = MADE(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part
def binarize(data, threshold=0.5):
    return torch.gt(data, threshold).float()  # 将大于阈值的部分设为1，小于等于阈值的部分设为0

binarize_transform = transforms.Lambda(lambda x: binarize(x))

transform = transforms.Compose([
    transforms.ToTensor(),
    binarize_transform
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
    batch_size=256,
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

loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Sample part, for compatibility with the Trainer
sample_valid = True
sample_valid_freq = 5   # sample images every 5 epochs


# Callbacks part
from callbacks import CheckpointResumeCallback, CheckpointSaveCallback

callbacks = [
    CheckpointResumeCallback(resume=True),
    CheckpointSaveCallback(every_n_epochs=1),
]