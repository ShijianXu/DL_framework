import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchmetrics.classification import MultilabelAUROC
from models.model_ecg_transformer import ECG_Transformer
from datasets.ptbxl_ecg_cls import PTBXL_CLS

# Model part
NUM_CLASSES = 5         # different tasks have different number of classes
ECG_CHANNELS = 12
NUM_LAYERS = 3
HIDDEN_SIZE = 128
NUM_HEADS = 8
CONTEXT_SIZE = 1000
EXPAND_SIZE = 256
ATTENTION = "multihead" # "multihead" or "causal" or "cross", etc.
ACTIVATE = nn.GELU           # activation function

# evaluation metric, can be predefined or customized or not provided
# depending on the model and the task
eval_metric = MultilabelAUROC(num_labels=NUM_CLASSES, average="macro", thresholds=None)


# feature_extractor = nn.Linear(ECG_CHANNELS, HIDDEN_SIZE)
# feature_extractor = nn.Conv1d(
#     in_channels=ECG_CHANNELS, 
#     out_channels=HIDDEN_SIZE, 
#     kernel_size=3, 
#     stride=2
# )

feature_extractor = nn.Sequential(
    nn.Conv1d(
        in_channels=ECG_CHANNELS, 
        out_channels=HIDDEN_SIZE, 
        kernel_size=3, 
        stride=2
    ),
    nn.ReLU(),
    nn.Conv1d(
        in_channels=HIDDEN_SIZE, 
        out_channels=HIDDEN_SIZE, 
        kernel_size=3, 
        stride=2
    ),
    nn.ReLU()
)

model_config = {
    "num_classes": NUM_CLASSES,
    "input_size": ECG_CHANNELS,
    "num_layers": NUM_LAYERS,
    "hidden_size": HIDDEN_SIZE,
    "num_heads": NUM_HEADS,
    "context_size": CONTEXT_SIZE,
    "expand_size": EXPAND_SIZE,
    "attention": ATTENTION,
    "act": ACTIVATE,
    "feature_extractor": feature_extractor,
}
model = ECG_Transformer(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part
TASK = "superdiagnostic"     # 'diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm', 'all'
SAMPLING_RATE = 100
DATA_PATH = "/home/xu0005/Desktop/ECG_data/ptb-xl/1.0.3/"

train_dataset = PTBXL_CLS(
    path=DATA_PATH,
    task=TASK,
    sampling_rate=SAMPLING_RATE,
    split="train"
)
# train_dataset, val_dataset = torch.utils.data.random_split(train_set, [50000, 10000])
train_dataloader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)
print("Construct train dataset with {} samples".format(len(train_dataset)))

valid_dataset = PTBXL_CLS(
    path=DATA_PATH,
    task=TASK,
    sampling_rate=SAMPLING_RATE,
    split="val"
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)
print("Construct valid dataset with {} samples".format(len(valid_dataset)))

test_dataset = PTBXL_CLS(
    path=DATA_PATH,
    task=TASK,
    sampling_rate=SAMPLING_RATE,
    split="test"
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

test_require_gt = True      # whether durign test requires ground truth
print("Construct test dataset with {} samples".format(len(test_dataset)))


# Loss and training part
num_epochs = 100
learning_rate = 1e-3

loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)


# Callbacks part
# The basic callbacks (resume and save) must be provided
# Otherwise, the training process will not be able to resumed or saved
from callbacks import CheckpointResumeCallback, CheckpointSaveCallback

callbacks = [
    CheckpointResumeCallback(resume=True),
    CheckpointSaveCallback(every_n_epochs=1),
]