import torch
import torch.nn as nn
import numpy as np

class SingleHeadAttention(nn.Module):
    """
    Single head bidirectional self-attention mechanism
    """
    def __init__(self,
                 hidden_size: int,
                 head_size: int,
                 bias: bool = True
    ):
        super().__init__()
        self.Wqkv = nn.Linear(hidden_size, 3 * head_size, bias=bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.head_size = head_size

    def forward(self, x):
        # batch size B, sequence length L, input dimension D
        # x: B x L x D

        qkv = self.Wqkv(x)  # B x L x 3 head_size
        q, k, v = qkv.chunk(3, dim=-1)

        attn = torch.einsum('bsc,btc->bst', q, k)   # QK^T
        attn = attn / np.sqrt(self.head_size)       # scale
        attn = torch.softmax(attn, dim=-1)          # softmax
        # print("attn shape: ", attn.shape)

        x = attn @ v                                # attention
        # print("x shape: ", x.shape)

        x = self.proj(x)                            # project back
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 attn_drop: float = 0.1,
                 proj_drop: float = 0.1,
                 bias: bool = True
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_size = hidden_size // num_heads
        self.num_heads = num_heads
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        # batch size B, sequence length L, input dimension D
        # x: B x L x D

        qkv = self.Wqkv(x)
        # print("qkv shape: ", qkv.shape)
        # B x L x 3 * hidden_size -> B x L x 3 x num_heads x head_size -> B x num_heads x 3 x L x head_size
        qkv = qkv.view(x.size(0), x.size(1), 3, self.num_heads, self.head_size).transpose(3, 1)
        # print("qkv shape: ", qkv.shape)
        q, k, v = qkv.unbind(dim=2)     # B x num_heads x L x head_size
        # print(q.shape, k.shape, v.shape)
        
        attn = torch.einsum('bnld,bnkd->bnlk', q, k)    # B x num_heads x L x L
        attn = attn / np.sqrt(self.head_size)           # scale

        attn = torch.softmax(attn, dim=-1)              # softmax
        
        attn = self.attn_drop(attn)                     # dropout

        x = attn @ v                                    # attention, B x num_heads x L x head_size
        # print("x shape: ", x.shape)
        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), -1) # B x L x D
        
        x = self.proj(x)                                # project back, B x L x D
        x = self.proj_drop(x)                           # dropout
        return x


class BidirectionalAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 attn_drop: float = 0.1,
                 proj_drop: float = 0.1,
                 bias: bool = True
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_size = hidden_size // num_heads
        self.num_heads = num_heads
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor):
        qkv = self.Wqkv(x)

        # B x L x 3 * hidden_size -> B x L x 3 x num_heads x head_size -> B x num_heads x 3 x L x head_size
        qkv = qkv.view(x.size(0), x.size(1), 3, self.num_heads, self.head_size).transpose(3, 1)
        q, k, v = qkv.unbind(dim=2)     # B x num_heads x L x head_size
        
        attn = torch.einsum('bnld,bnkd->bnlk', q, k)    # B x num_heads x L x L
        attn = attn / np.sqrt(self.head_size)           # scale

        if mask is not None:
            # mask shape: B x L -> B x 1 x 1 x L
            attn = attn.masked_fill(mask.view(x.size(0), 1, 1, x.size(1)), float('-inf'))

        attn = torch.softmax(attn, dim=-1)              # softmax        
        attn = self.attn_drop(attn)                     # dropout

        x = attn @ v                                    # attention, B x num_heads x L x head_size

        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), -1) # B x L x D
        x = self.proj(x)                                # project back, B x L x D
        x = self.proj_drop(x)                           # dropout
        return x
    

class CausalAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 context_size: int,
                 attn_drop: float = 0.1,
                 proj_drop: float = 0.1,
                 bias: bool = True
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_size = hidden_size // num_heads
        self.num_heads = num_heads
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.register_buffer('causal_mask',
            torch.triu(torch.ones([context_size, context_size],
                       dtype=torch.bool), diagonal=1)
                .view(1, 1, context_size, context_size))

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor):
        qkv = self.Wqkv(x)

        # B x L x 3 * hidden_size -> B x L x 3 x num_heads x head_size -> B x num_heads x 3 x L x head_size
        qkv = qkv.view(x.size(0), x.size(1), 3, self.num_heads, self.head_size).transpose(3, 1)
        q, k, v = qkv.unbind(dim=2)     # B x num_heads x L x head_size
        
        attn = torch.einsum('bnld,bnkd->bnlk', q, k)    # B x num_heads x L x L
        attn = attn / np.sqrt(self.head_size)           # scale

        if mask is not None:
            # mask shape: B x L -> B x 1 x 1 x L
            combined_mask = mask.view(x.size(0), 1, 1, x.size(1)) + self.causal_mask
            attn = attn.masked_fill(combined_mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)              # softmax        
        attn = self.attn_drop(attn)                     # dropout

        x = attn @ v                                    # attention, B x num_heads x L x head_size

        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), -1) # B x L x D
        x = self.proj(x)                                # project back, B x L x D
        x = self.proj_drop(x)                           # dropout
        return x

class CrossAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 context_size: int,
                 attn_drop: float = 0.1,
                 proj_drop: float = 0.1,
                 bias: bool = True
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_size = hidden_size // num_heads
        self.num_heads = num_heads
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.Wkv = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.register_buffer('causal_mask',
            torch.triu(torch.ones([context_size, context_size],
                       dtype=torch.bool), diagonal=1)
                .view(1, 1, context_size, context_size))

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.BoolTensor):
        # B x L x hidden_size -> B x L x num_heads x head_size -> B x num_heads x L x head_size
        q = self.Wq(x).view(x.size(0), x.size(1), self.num_heads, self.head_size).transpose(1, 2)

        # B x L x 2 * hidden_size -> B x L x 2 x num_heads x head_size -> B x num_heads x 2 x L x head_size
        kv = self.Wkv(y).view(y.size(0), y.size(1), 2, self.num_heads, self.head_size).transpose(3, 1)
        k, v = kv.unbind(dim=2)     # B x num_heads x L x head_size

        # compute attention
        attn = torch.einsum('bnld,bnkd->bnlk', q, k)    # B x num_heads x L x L
        attn = attn / np.sqrt(self.head_size)           # scale

        if mask is not None:
            # mask shape: B x L -> B x 1 x 1 x L
            combined_mask = mask.view(x.size(0), 1, 1, x.size(1)) + self.causal_mask
            attn = attn.masked_fill(combined_mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)              # softmax        
        attn = self.attn_drop(attn)                     # dropout

        x = attn @ v                                    # attention, B x num_heads x L x head_size

        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), -1) # B x L x D
        x = self.proj(x)                                # project back, B x L x D
        x = self.proj_drop(x)                           # dropout
        return x
    

class FeedForward(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 expand_size: int,
                 act: nn.Module = nn.GELU,
                 dropout: float = 0.1,
                 bias: bool = True
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, expand_size, bias=bias)
        self.fc2 = nn.Linear(expand_size, hidden_size, bias=bias)
        self.act = act()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.act(self.fc1(x))
        
        x = self.fc2(x)

        x = self.drop(x)        # it can be placed before or after the the second linear layer
        return x


class TransformerBlock(nn.Module):
    def __init__(self,
        hidden_size: int,
        num_heads: int,
        context_size: int,
        expand_size: int,
        attention: str = 'causal',
        act: nn.Module = nn.GELU,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        ffn_drop: float = 0.1,
        bias: bool = True,
        pre_norm: bool = False
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        if attention == 'causal':
            self.attn = CausalAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                context_size=context_size,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                bias=bias
            )
        elif attention == 'cross':
            self.attn = CrossAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                context_size=context_size,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                bias=bias
            )
        elif attention == 'multihead':
            self.attn = MultiHeadAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                bias=bias
            )
        else:
            raise ValueError("Attention type not supported")

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            expand_size=expand_size,
            act=act,
            dropout=ffn_drop,
            bias=bias
        )

        self.pre_norm = pre_norm

    def forward(self, x):
        if self.pre_norm:
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x +self.attn(x))
            x = self.norm2(x + self.ffn(x))

        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self,
        context_size: int,
        hidden_size: int
    ):
        super().__init__()
        # create the positional encoding tensor of shape
        # maximum sequence length by embedding dimension
        pe = torch.zeros(context_size, hidden_size, dtype=torch.float)

        # pre-populate the position and the div_terms
        position = torch.arange(context_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-np.log(10000) / hidden_size)
        )

        # even positional encodings use sine, odd cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as a buffer so autograd doesn't modify
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # return the pre-calculated positional encodings
        # up to sequence length (L). output shape (1, L, hidden_size)
        return self.pe[:, :x.shape[1], :]



if __name__ == "__main__":
    x = torch.randn(32, 10, 512)            # embedded token
    # att = SingleHeadAttention(512, 64)
    att = CrossAttention(512, 8)
    y = torch.randn(32, 10, 512)            # embedded token
    mask = torch.zeros(32, 10).bool()
    out = att(x, y, mask)
    print(out.shape)