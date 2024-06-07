# test of CNF on Circle data

import torch
from torch.utils.data import DataLoader

from models.model_simple_cnf import SimpleCNF
from datasets.circle_sklearn import CircleDataset
from losses.losses import NLLLoss_CNF

# Model part
model_config = {
    "in_out_dim": 2, 
    "hidden_dim": 32, 
    "width": 64, 
    "t0": 0, 
    "t1": 10
}
model = SimpleCNF(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part

train_dataset = CircleDataset(num_samples=512*1000)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=2,
    drop_last=True
)
print("Construct train dataset with {} samples".format(len(train_dataset)))

# Loss and training part
num_epochs = 1
learning_rate = 1e-3

loss = NLLLoss_CNF()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# Callbacks part
from callbacks import CheckpointResumeCallback, CheckpointSaveCallback, TrainingTimerCallback

callbacks = [
    CheckpointResumeCallback(resume=True),
    CheckpointSaveCallback(every_n_epochs=1),
    TrainingTimerCallback()
]