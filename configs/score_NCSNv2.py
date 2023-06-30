import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import LSUN

import models.model_ncsn_v2
from losses import Anneal_DSM_Loss

# Model Part
NOISE_LEVELS = 788
IMG_SIZE = 64
CHANNELS = 3

backbone = {}
backbone["name"] = "models.backbone_ncsnv2.NCSNv2_RefineNet"
backbone["config"] = {
    "in_channels": CHANNELS,
    "ngf": 128,
    "num_classes": NOISE_LEVELS,
    "img_size": IMG_SIZE,
    "norm": "InstanceNorm++",
    "nonlinearity": "elu",
}

backend = None

model_config = {
    "backbone": backbone,
    "backend": backend,
    "backbone_created": False,
    "noise_levels": NOISE_LEVELS,
    "sigma_begin": 140,
    "sigma_end": 0.01,
    "sigma_dist": "geometric",
    "ema": True,
    "ema_rate": 0.999,

    # Sampling Part
    "step_lr": 0.0000049,
    "n_steps_each": 4,
}

model = models.model_ncsn_v2.NCSNv2(**model_config)
print(f"Total model parameterrs: {model.get_num_params()}")


# Dataset Part
BATCH_SIZE = 32
train_folder = 'church_outdoor_train'
val_folder = 'church_outdoor_val'
dataset = LSUN(root=os.path.join('./data/lsun'), classes=[train_folder],
                transform=transforms.Compose([
                    transforms.Resize(IMG_SIZE),
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
            ]))

test_dataset = LSUN(root=os.path.join('./data/lsun'), classes=[val_folder],
                transform=transforms.Compose([
                    transforms.Resize(IMG_SIZE),
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.ToTensor(),
            ]))

print("Dataset Size: ", len(dataset))
print("Test Dataset Size: ", len(test_dataset))

train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

valid_dataloader = None


# Optimizer Part
num_epochs = 300
learning_rate = 0.0001
loss = Anneal_DSM_Loss(anneal_power=2)
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=learning_rate, 
    betas=(0.9, 0.999),
    amsgrad=False,
    eps=1e-8
)

scheduler = None
sample_valid = True
sample_valid_freq = 5   # sample valid every 5 epochs
