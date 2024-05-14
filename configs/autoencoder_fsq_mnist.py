import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from models.model_simple_fsq_ae import SimpleFSQAutoEncoder

# Model part
model_config = {
    "levels": [8, 6, 5]
}
model = SimpleFSQAutoEncoder(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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
    shuffle=True
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
    shuffle=False
)
print("Construct test dataset with {} samples".format(len(test_dataset)))


# Loss and training part
num_epochs = 500
learning_rate = 3e-4

loss = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Callbacks part
from callbacks import CheckpointResumeCallback, CheckpointSaveCallback

callbacks = [
    CheckpointResumeCallback(resume=True),
    CheckpointSaveCallback(every_n_epochs=1),
]