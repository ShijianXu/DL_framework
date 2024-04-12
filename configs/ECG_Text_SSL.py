import torch
from torch.utils.data import DataLoader
from datasets.ptbxl_dataset import PTBXL
from transformers import BertTokenizer
import models.frozenLM
import models.model_ecg_text
from models.resnet1d import BasicBlock1D, ResNet1D
from losses import ContrastiveLoss

# Model Part
pretrained_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
text_encoder = models.frozenLM.FrozenLM(pretrained_model_name)
embedding_dim = text_encoder.language_model.config.hidden_size


# TODO: how to get the in_channels for the ecg_encoder?
ecg_encoder = ResNet1D(
    in_channels=1000, 
    block=BasicBlock1D,
    layers=[2, 2, 2, 2],
    projection_size=embedding_dim
)

model_config = {
    "tokenizer": tokenizer,
    "text_encoder": text_encoder,
    "ecg_encoder": ecg_encoder,
}

model = models.model_ecg_text.ECG_Text_CLIP(**model_config)
print(f"Total model parameters: {model.get_num_params()}")

# Dataset Part
train_dataset = PTBXL(
    path='/home/xu0005/Desktop/ECG_data/ptb-xl/1.0.3/', 
    sampling_rate=100, 
    train=True
)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=2
)
print("Construct train dataset with {} samples".format(len(train_dataset)))

valid_dataset = PTBXL(
    path='/home/xu0005/Desktop/ECG_data/ptb-xl/1.0.3/', 
    sampling_rate=100, 
    train=False
)
# valid_dataloader = DataLoader(
#     valid_dataset, 
#     batch_size=32, 
#     shuffle=False,
#     num_workers=2
# )
valid_dataloader = None


# Loss and training part
num_epochs = 100
learning_rate = 0.001
loss = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=0.0
                             )
scheduler = None

# below two are for generative model sampling,
# does not needed here, just for compatibility
sample_valid = False
sample_valid_freq = -1