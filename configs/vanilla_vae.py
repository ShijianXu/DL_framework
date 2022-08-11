import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.celeba_dataset import MyCelebA
from losses import VAE_Loss
import models.model_vanilla_vae

# Model part
backbone = {}
backbone["name"] = "models.backbone_vae_encoder.VAE_Encoder"
backbone["config"] = {
    "in_channels": 3, 
    "latent_dim": 128,
    "hidden_dims": [32, 64, 128, 256, 512]
}

backend = {}
backend["name"] = "models.backend_vae_decoder.VAE_Decoder"
backend["config"] = {
    "out_channels": 3, 
    "latent_dim": 128,
    "hidden_dims": [32, 64, 128, 256, 512]
}

model_config = {
    "backbone": backbone,
    "backend": backend,
    "backbone_created": False,
    "backend_created": False,
}
model = models.model_vanilla_vae.VanillaVAE(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(148),
    transforms.Resize(64),
    transforms.ToTensor(),
])
train_dataset = MyCelebA(
    './data/', 
    split='train',
    transform=train_transforms,
    download=False,
)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=2
)
print("Construct train dataset with {} samples".format(len(train_dataset)))


valid_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(148),
    transforms.Resize(64),
    transforms.ToTensor(),
])
valid_dataset = MyCelebA(
    './data/', 
    split='test',
    transform=valid_transforms,
    download=False,
)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=1
)
print("Construct test dataset with {} samples".format(len(valid_dataset)))


# Test dataset & test dataloader (not paired, just some corrupted inputs)
test_dataset = MyCelebA(
    './data/', 
    split='test',
    transform=valid_transforms,
    download=False,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)
test_require_gt = False


# Loss and training part
num_epochs = 500
learning_rate = 0.001
loss = VAE_Loss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate, 
                             betas=(0.9, 0.99),
                             eps=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                patience=num_epochs/4, factor=0.5, verbose=True)