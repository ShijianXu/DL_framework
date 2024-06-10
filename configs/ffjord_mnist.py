
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.model_ffjord import FFJORD
from losses.losses import NLLLoss_CNF


# Model part
model_config = {
    "input_shape": (1, 28, 28),
    "hidden_dims": [8,32,32,8],
    "strides": [2,2,1,-2,-2],
    "t0": 0,
    "t1": 1,
    "hutchinson_trace": True
}
model = FFJORD(**model_config)
print(f"Total model parameters: {model.get_num_params()}")

# Data part

def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    noise = x.new().resize_as_(x).uniform_()
    x = x * 255 + noise
    x = x / 256
    return x

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    add_noise
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
    batch_size=200,
    shuffle=False,
    drop_last=True
)
print("Construct test dataset with {} samples".format(len(test_dataset)))

sample_valid = True
sample_valid_freq = 5


# Loss and training part
num_epochs = 200
loss = NLLLoss_CNF()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Callbacks part
from callbacks import CheckpointResumeCallback, CheckpointSaveCallback, TrainingTimerCallback

callbacks = [
    CheckpointResumeCallback(resume=True),
    CheckpointSaveCallback(every_n_epochs=1),
    TrainingTimerCallback()
]