import torch
from torch.utils.data import DataLoader

from datasets.n2n_dataset import Noise2NoiseDataset
import models.model_noise2noise

# Model part
backbone = {}
backbone["name"] = "models.backbone_unet_n2n.UNet"
backbone["config"] = {
    "in_channels": 3,
    "out_channels": 3
}

backend = torch.nn.Identity()
model_config = {
    "backbone": backbone,
    "backend": backend,
    "backbone_created": False,
    "backend_created": True,
}

model = models.model_noise2noise.Noise2Noise(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part
train_dataset = Noise2NoiseDataset('./data/DIV2K_train_80', crop_size=64)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=2
)
print("Construct train dataset with {} samples".format(len(train_dataset)))

valid_dataset = Noise2NoiseDataset('./data/DIV2K_valid_20', crop_size=64, clean_targets=True)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=1
)
print("Construct test dataset with {} samples".format(len(valid_dataset)))


# Test dataset & test dataloader (not paired, just some corrupted inputs)
# test_dataset = xxx
# test_dataloader = DataLoader()


# Loss and training part
num_epochs = 10
learning_rate = 0.001
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate, 
                             betas=(0.9, 0.99),
                             eps=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                patience=num_epochs/4, factor=0.5, verbose=True)