import torch

def create_padding_mask(seq, pad_idx=0):
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.uint8)
    return mask == 1

def create_masks(src, tgt, pad_idx=0):
    src_mask = create_padding_mask(src, pad_idx)
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)
    look_ahead_mask = create_look_ahead_mask(tgt.size(1))
    combined_mask = torch.max(tgt_padding_mask, look_ahead_mask.to(tgt.device))
    return src_mask, combined_mask

def train_step(model, optimizer, loss_fn, src, tgt_inp, tgt_real):
    model.train()
    src_mask, tgt_mask = create_masks(src, tgt_inp)
    preds = model(src, tgt_inp, src_mask, tgt_mask)
    loss = loss_fn(preds.view(-1, preds.size(-1)), tgt_real.view(-1))
    optimizer.optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
