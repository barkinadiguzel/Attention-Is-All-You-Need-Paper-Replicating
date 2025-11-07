import torch
import torch.nn as nn
from src.4_encoder_decoder.encoder import Encoder
from src.4_encoder_decoder.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
        output = self.linear(dec_output)
        output = self.softmax(output)
        return output
