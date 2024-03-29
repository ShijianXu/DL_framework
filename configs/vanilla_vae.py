import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.celeba_dataset import MyCelebA, TinyCelebA
from losses import VAE_Loss
import models.model_vanilla_vae

# Model part
latent_dim = 128

backbone = {}
backbone["name"] = "models.backbone_vae_encoder.VAE_Encoder"
backbone["config"] = {
    "in_channels": 3, 
    "latent_dim": latent_dim,
    "hidden_dims": [32, 64, 128, 256, 512]
}

backend = {}
backend["name"] = "models.backend_vae_decoder.VAE_Decoder"
backend["config"] = {
    "out_channels": 3, 
    "latent_dim": latent_dim,
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
train_dataset = TinyCelebA(
    './data/celeba/img_align_celeba',
    sample_nums=10000,
    transform=train_transforms
)
# train_dataset = MyCelebA(
#     './data/', 
#     split='train',
#     transform=train_transforms,
#     download=False,
# )
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=2
)
print("Construct train dataset with {} samples".format(len(train_dataset)))


valid_transforms = transforms.Compose([
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
    batch_size=4, 
    shuffle=False, 
    num_workers=2
)
print("Construct test dataset with {} samples".format(len(valid_dataset)))


# Loss and training part
valid_sample = True     # sample after each epoch validation
num_epochs = 100
learning_rate = 0.005
loss = VAE_Loss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=0.0
                             )
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)