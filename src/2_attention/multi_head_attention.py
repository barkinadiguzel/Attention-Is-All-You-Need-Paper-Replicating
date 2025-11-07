class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WO = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # linear projection ve reshape
        Q = self.WQ(Q).view(batch_size, -1, self.h, self.d_k).transpose(1,2)
        K = self.WK(K).view(batch_size, -1, self.h, self.d_k).transpose(1,2)
        V = self.WV(V).view(batch_size, -1, self.h, self.d_k).transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

        output, attn = self.attention(Q, K, V, mask)

        # concat headler
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        output = self.WO(output)
        return output, attn
