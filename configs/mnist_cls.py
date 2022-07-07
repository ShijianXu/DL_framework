import torch
import torchvision

backbone = {}
backbone["name"] = "resnet"

last_module = {}
last_module["name"] = "classifier"

model_config = {}

model = torchvision.models.resnet18(pretrained=False)


train_dataset = None
train_dataloader = None

print("Model init.")