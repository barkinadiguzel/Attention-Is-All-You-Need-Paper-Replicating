import torch

def loss_fn(output, target, pad_idx=0, smoothing=0.1):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=smoothing)
    loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
    return loss
