import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from models.model_conv_res_ode import ConvResODE 


# Model part
model_config = {
    "num_classes": 10,
    "in_channels": 1,
    "num_filters": 64,
    "downsampling_method": 'conv',
    "is_odenet": True
}
model = ConvResODE(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part
transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    'data', 
    train=True, 
    download=True, 
    transform=transform_train
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    drop_last=True
)
print("Construct train dataset with {} samples".format(len(train_dataset)))


transform_test = transforms.Compose([
    transforms.ToTensor(),
])

valid_dataloader = DataLoader(
    datasets.MNIST(root='data', train=True, download=True, transform=transform_test),
    batch_size=1000, shuffle=False, num_workers=2, drop_last=True
)

test_dataset = datasets.MNIST(
    'data',
    train=False,
    download=True,
    transform=transform_test
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=2,
    drop_last=True
)
print("Construct test dataset with {} samples".format(len(test_dataset)))


# Loss and training part
num_epochs = 160
learning_rate = 0.1

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

milestones=[60, 100, 140]
decay_rates=[1, 0.1, 0.01, 0.001]
gamma = decay_rates[-1]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Callbacks part
from callbacks import CheckpointResumeCallback, CheckpointSaveCallback

callbacks = [
    CheckpointResumeCallback(resume=True),
    CheckpointSaveCallback(every_n_epochs=1),
]