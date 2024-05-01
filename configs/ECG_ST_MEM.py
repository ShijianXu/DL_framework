import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from models.model_autogressive_MADE import MADE

# Model part



model_config = {
}
model = MADE(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part
train_dataset = None

# train_dataset, val_dataset = torch.utils.data.random_split(train_set, [50000, 10000])
train_dataloader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=2
)
print("Construct train dataset with {} samples".format(len(train_dataset)))

valid_dataloader = None

test_dataset = None
print("Construct test dataset with {} samples".format(len(test_dataset)))


# Loss and training part
num_epochs = 500
learning_rate = 1e-3

loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Sample part, for compatibility with the Trainer
sample_valid = False
sample_valid_freq = -1   # sample images every 5 epochs


# Callbacks part
from callbacks import CheckpointResumeCallback, CheckpointSaveCallback

callbacks = [
    CheckpointResumeCallback(resume=True),
    CheckpointSaveCallback(every_n_epochs=1),
]