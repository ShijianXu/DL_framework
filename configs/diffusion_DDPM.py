import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from losses import Diffusion_Loss
import models.model_diffusion_DDPM

# Model part
IMG_SIZE = 64

backbone = {}
# backbone["name"] = "models.backbone_diffusion_unet.SimpleUnet"
# backbone["config"] = {
#     "image_channels": 3,
#     "down_channels": [64, 128, 256, 512, 1024],
#     "up_channels": [1024, 512, 256, 128, 64],
#     "out_dim": 3,
#     "time_emb_dim": 32
# }

backbone["name"] = "models.backbone_diffusion_convnext.Unet"
backbone["config"] = {
    "dim": IMG_SIZE,
    "use_convnext": False,
}


backend = None

model_config = {
    "backbone": backbone,
    "backend": backend,
    "backbone_created": False,
    "T": 200,
}
model = models.model_diffusion_DDPM.DiffusionDDPM(**model_config)
print(f"Total model parameterrs: {model.get_num_params()}")


# Data part
BATCH_SIZE = 128

def load_transformed_dataset():
    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                     # convert to [0,1]
        transforms.Lambda(lambda x: x * 2. - 1.),  # convert to [-1,1]
    ])

    train = torchvision.datasets.StanfordCars(root='./data', download=True, 
                                              transform=data_transform, split='train')
    test = torchvision.datasets.StanfordCars(root='./data', download=True,
                                                transform=data_transform, split='test')
    
    return torch.utils.data.ConcatDataset([train, test])

dataset = load_transformed_dataset()
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True)
print("Concate train and test datasets with total {} samples".format(len(dataset)))

valid_dataloader = None   # No validation dataset, but we need to define it


# Loss abd training part
num_epochs = 100
learning_rate = 1e-3
loss = Diffusion_Loss(loss_type="l1")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = None
sample_valid = True
sample_valid_freq = 5       # every 5 epochs generate images for validation