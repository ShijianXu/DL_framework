import torch
import torch.nn as nn
import utils
from .module_transformer import *

class ECG_Transformer(nn.Module):
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

        self.feature_extractor = feature_extractor
        self.pos_embed = nn.Embedding(context_size, hidden_size)
        self.embed_drop = nn.Dropout(embed_drop)

        # initialize num_layers of transformer layers
        self.tfm_blocks = nn.ModuleList([TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                context_size=context_size, 
                expand_size=expand_size,
                attention=attention,
                act=act,
                bias=bias,
                attn_drop=attn_drop, 
                proj_drop=out_drop,
                ffn_drop=ffn_drop)
            for _ in range(num_layers)])

        # optional pre-head normalization
        if head_norm:
            self.head_norm = nn.LayerNorm(hidden_size)
        else:
            self.head_norm = nn.Identity()

        # predicts the class for the input sequence
        self.head = nn.Linear(hidden_size, num_classes, bias=head_bias)

        # precreate positional indices for the positional embedding
        pos = torch.arange(0, context_size, dtype=torch.long)
        self.register_buffer('pos', pos, persistent=False)

        self.apply(self._init_weights)

        # for test/val metric
        self.metric_m = utils.AverageMeter()
        self.best_metric = 0

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels), B x S x C
        # the final input length L does not need to be the same as the input sequence length S
        # convolutional may reduce the length, and the transformer layers can handle variable length
        # shorter lengths will cost less computation in the attention mechanism
        if isinstance(self.feature_extractor, nn.Linear):
            x = self.feature_extractor(x)               # B x L x hidden_size
        else:
            x = self.feature_extractor(x.transpose(1, 2)).transpose(1, 2)


        pos = self.pos_embed(self.pos[:x.shape[1]])     # L x hidden_size

        x = self.embed_drop(x + pos)                    # B x L x hidden_size

        for tfm_block in self.tfm_blocks:
            x = tfm_block(x)                            # B x L x hidden_size

        x = self.head_norm(x)                           # B x L x hidden_size

        # average pool over the sequence dimension
        x = x.mean(dim=1)                               # B x hidden_size

        x = self.head(x)                                # B x num_classes
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module._get_name() == 'fc2':
                # GPT-2 style FFN init
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/np.sqrt(2 * self.num_layers))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def compute_loss(self, output, target, criterion):
        loss = criterion(output, target)
        return {"loss": loss}

    def process_batch(self, batch, criterion, device):
        ecg_data, target = batch[0].to(device), batch[1].to(device)
        output = self(ecg_data)
        loss = self.compute_loss(output, target, criterion)
        return loss
    
    def compute_metric(self, source, preds, target, eval_metric):
        metric_value = eval_metric(preds, target.int())
        self.metric_m.update(metric_value)

    def get_metric_value(self):
        return self.metric_m.avg
    
    def is_best_metric(self):
        if self.metric_m.avg > self.best_metric:
            self.best_metric = self.metric_m.avg
            return True
        else:
            return False

    def display_metric_value(self):
        print(f'AUC value: {self.get_metric_value()}')

    def reset_metric(self):
        self.metric_m.reset()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def everything_to(self, device):
        pass



if __name__ == "__main__":
    feature_extractor = nn.Conv1d(
        in_channels=12, 
        out_channels=256, 
        kernel_size=3, 
        stride=2
    )

    # model = ECG_Transformer(
    #     num_classes=2,
    #     input_size=12,
    #     num_layers=3,
    #     hidden_size=256,
    #     num_heads=8,
    #     context_size=5000,
    #     expand_size=512,
    #     feature_extractor=feature_extractor,
    #     attention="multihead",
    #     act=nn.GELU,
    #     embed_drop=0.1,
    #     attn_drop=0.1,
    #     out_drop=0.1,
    #     ffn_drop=0.1,
    #     head_norm=True,
    #     head_bias=True,
    #     bias=True
    # )
    x = torch.randn(32, 1000, 12)
    y = feature_extractor(x.transpose(1, 2)).transpose(1, 2)
    print(y.shape)