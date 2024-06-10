
import torch
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np

from models.model_ffjord import FFJORD
from datasets.dataset_single_image import DatasetfromImage
from losses.losses import NLLLoss_CNF


# Model part
model_config = {
    "input_shape": (2,),
    "hidden_dims": [64, 64, 64],
    "t0": 0,
    "t1": 1
}
model = FFJORD(**model_config)
print(f"Total model parameters: {model.get_num_params()}")

# Data part
img_path = '/home/xu0005/Desktop/github.png'
raw_img = np.array(Image.open(img_path).convert('L'))
# resize the image to 100x100
resized_img = Image.fromarray(raw_img).resize((100, 100), Image.LANCZOS)

# Convert the PIL image to a numpy array
resized_img = np.array(resized_img)

pad = 50
img = np.zeros((resized_img.shape[0]+2*pad,resized_img.shape[1]+2*pad))
img[pad:-pad,pad:-pad] = resized_img[:,:]


train_dataset = DatasetfromImage(img, max_val=4.0, sample_size=40000)
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
print("Construct train dataset with {} samples".format(len(train_dataset)))


# Loss and training part
num_epochs = 150
loss = NLLLoss_CNF()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Callbacks part
from callbacks import CheckpointResumeCallback, CheckpointSaveCallback, TrainingTimerCallback

callbacks = [
    CheckpointResumeCallback(resume=True),
    CheckpointSaveCallback(every_n_epochs=1),
    TrainingTimerCallback()
]