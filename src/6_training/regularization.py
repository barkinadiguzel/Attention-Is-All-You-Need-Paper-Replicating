import torch.nn as nn

def apply_dropout(x, rate=0.1):
    return nn.Dropout(rate)(x)
