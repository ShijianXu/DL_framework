import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
# from models.backbone_unet_diffusion import SimpleUnet
# from losses import Diffusion_Loss_fn

#=================================================================================================
# Unet
from torch import nn
import math

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First conv
        h = self.bnorm1(self.relu(self.conv1(x)))

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))

        # Extend last 2 dims
        time_emb = time_emb[(..., ) + (None, ) * 2]

        # Add time channel
        h = h + time_emb

        # Second conv
        h = self.bnorm2(self.relu(self.conv2(h)))

        # Down or Upsample
        h = self.transform(h)

        return h
    

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        return embeddings
    

class SimpleUnet(nn.Module):
    """
    A simplified variant of the U-Net architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = [64, 128, 256, 512, 1024]
        up_channels = [1024, 512, 256, 128, 64]
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim) \
                for i in range(len(down_channels) - 1)
        ])

        # Upsample
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) \
                for i in range(len(up_channels) - 1)
        ])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embed timestep
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()

            x = torch.cat([x, residual_x], dim=1)
            x = up(x, t)

        return self.output(x)
    
#=================================================================================================
# Loss
def Diffusion_Loss_fn(model, x_0, t, loss_type="l1"):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device=x_0.device)
    noise_pred = model(x_noisy, t)
    
    if loss_type == 'l1':
        loss = F.l1_loss(noise, noise_pred)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, noise_pred)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, noise_pred)
    else:
        raise NotImplementedError()

    return loss


#=================================================================================================

def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset. """
    plt.figure(figsize=(8, 8))
    for i, img in enumerate(dataset):
        plt.subplot(num_samples // cols, cols, i + 1)
        plt.imshow(img[0])
        # plt.axis('off')
        if i + 1 == num_samples:
            break

    plt.show()


# data = torchvision.datasets.StanfordCars(root='./data', download=True)
# show_images(data)


#========================================
import torch.nn.functional as F

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a list of values vals
    while considering the batch dimension of x_shape.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape)-1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and 
    returns the nosiy image for that timestep.
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cimprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    # mean + variance
    # x_t= \sqrt \bar alpha_t * x_0 + \sqrt(1-\bar alpha_t) * N(0,1)
    return sqrt_alphas_cumprod_t.to(device) * x_0 + \
        sqrt_one_minus_alphas_cimprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 200
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)                      # \bar alpha_t = \prod_{i=1}^t (1-\alpha_i)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # \bar alpha_{t-1}
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)                    # sqrt(\bar alpha_t)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)     # sqrt(1- \bar alpha_t)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# Prepare the data
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

IMG_SIZE = 64
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


def show_tensor_image(image):
    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1.) / 2.),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),            # [C, H, W] -> [H, W, C]
        transforms.Lambda(lambda x: x * 255),                       # [0,1] -> [0,255]
        transforms.Lambda(lambda x: x.numpy().astype(np.uint8)),    # convert to numpy
        transforms.ToPILImage(),                                    # convert to PIL image
    ])

    # take first image from batch
    if len(image.shape) == 4:
        image = image[0]
    plt.imshow(reverse_transform(image))


data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# Simulate forward diffusion
# image = next(iter(dataloader))[0]
# plt.figure(figsize=(15, 6))
# plt.axis('off')
# num_images = 10
# stepsize = int(T / num_images)

# for idx in range(0, T, stepsize):
#     t = torch.Tensor([idx]).type(torch.int64)   # timestep
#     plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
#     img, noise = forward_diffusion_sample(image, t)
#     show_tensor_image(img)
    
# plt.show()
# why the image at idx=0 also has noise?


model = SimpleUnet()

#====================================================================================
# Sampling

@torch.no_grad()
def sample_timestep(x, t):
    """
    Call the model to predict the noise in the image x_t at time t,
    get the clean image x_t-1.
    Apply extra noise to the denoised image, unless we are at the t=0.
    """
    betas_t = get_index_from_list(betas, t, x.shape)            # \beta_t
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(      # sqrt(1- \bar alpha_t)
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(                  # \sqrt(1/alpha_t)
        sqrt_recip_alphas, t, x.shape
    )

    # predict noise
    noise_pred = model(x, t)

    # get the mean: x_t - noise_pred
    model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image():
    # sample noise
    img_size = IMG_SIZE

    # img start from pure noise
    img = torch.randn((1, 3, img_size, img_size), device=device)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, dtype=torch.int64, device=device)
        img = sample_timestep(img, t)
        
        img = torch.clamp(img, -1., 1.)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize) + 1)
            show_tensor_image(img.detach().cpu())

    plt.show()


@torch.no_grad()
def log_images(epoch):
    # sample noise
    img_size = IMG_SIZE

    # img start from pure noise
    img = torch.randn((1, 3, img_size, img_size), device=device)

    num_images = 10
    stepsize = int(T / num_images)
    image_list = []

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, dtype=torch.int64, device=device)
        img = sample_timestep(img, t)
        
        img = torch.clamp(img, -1., 1.)
        if i % stepsize == 0:
            image_list.append((img + 1.) / 2.)  # convert to [0,1]

    # Create a grid of images
    grid = vutils.make_grid(image_list, nrow=num_images, normalize=True, scale_each=True)

    # Log the grid to TensorBoard
    writer.add_image(f'Sampled Images at epoch: {epoch}', grid, global_step=epoch)


#====================================================================================
# Training
from torch.optim import Adam

writer = SummaryWriter(log_dir='./logs/diffusion')

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model.to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
epochs = 100

print("Start training...")
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        batch = batch.to(device)

        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = Diffusion_Loss_fn(model, batch[0], t, forward_diffusion_sample)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 and step == 0:
            print(f"Epoch {epoch} | Loss {loss.item()}")
            # sample_plot_image()
            log_images(epoch)