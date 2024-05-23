import torch
import torch.nn as nn
import utils
from .module_transformer import *

class ECG_Mixed_Regression(nn.Module):
    def __init__(self,
        num_classes:int,
        input_size:int,
        num_layers:int,
        hidden_size:int,
        num_heads:int,
        context_size:int,
        expand_size:int,
        feature_extractor:nn.Module,
        attention:str="multihead",
        act: nn.Module = nn.GELU,
        embed_drop:float=0.1,
        attn_drop:float=0.1,
        out_drop:float=0.1,
        ffn_drop:float=0.1,
        head_norm:bool=True,
        head_bias:bool=True,
        bias:bool=True,
    ):
        super().__init__()
        

