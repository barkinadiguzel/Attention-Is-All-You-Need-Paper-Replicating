import torch
import torch.nn as nn
from src.attention.multi_head_attention import MultiHeadAttention
from src.feed_forward.positionwise_ffn import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, h=num_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        _x = x
        attn_output, _ = self.self_attn(x, x, x, mask=mask)
        x = _x + self.dropout1(attn_output)
        x = self.norm1(x)
        _x = x
        ffn_output = self.ffn(x)
        x = _x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x
