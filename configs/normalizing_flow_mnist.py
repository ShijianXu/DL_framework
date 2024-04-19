import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from losses import RealNVPLoss

from models.model_normalizing_flow_image import Image_NormalizingFlow
from models.module_coupling_layer import *
from models.module_dequantization import Dequantization, VariationalDequantization
from models.module_gatedresnet import GatedConvNet

# Model part
IMG_SIZE = (1, 28, 28)
use_vardeq = True
flow_layers = []
if use_vardeq:
    vardeq_layers = [AffineCouplingLayer(
                        net=GatedConvNet(c_in=2, c_out=2, c_hidden=16),
                        in_channels=1,      # for MNIST, we only have 1 channel
                        mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1))
                        ) for i in range(4)]
    flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]
else:
    flow_layers += [Dequantization()]

for i in range(8):
    flow_layers += [AffineCouplingLayer(
                        net=GatedConvNet(c_in=1, c_hidden=32),
                        in_channels=1,
                        mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1))
                        )]

model_config = {
    "flows": flow_layers,
    "prior": torch.distributions.Normal(loc=0, scale=1) # Initialize the prior distribution as a standard normal distribution
}
model = Image_NormalizingFlow(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part
# Discretize the image to 0-255
# why ???
def discretize_image(image):
    return (image * 255).to(torch.int32)

transform = transforms.Compose([
    transforms.ToTensor(),
    discretize_image
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
    batch_size=128,
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
num_epochs = 100
learning_rate = 1e-4

loss = RealNVPLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Sample part, for compatibility with the Trainer
sample_valid = True
sample_valid_freq = 5   # sample images every 5 epochs