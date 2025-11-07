import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q: Queries  -> shape: [batch_size, heads, seq_len_q, d_k]
        K: Keys     -> shape: [batch_size, heads, seq_len_k, d_k]
        V: Values   -> shape: [batch_size, heads, seq_len_v, d_v]
        mask: optional tensor to block certain positions (e.g., decoder masking)
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output, attn
