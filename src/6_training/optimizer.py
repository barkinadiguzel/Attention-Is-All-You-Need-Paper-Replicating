import torch
import math

def get_noam_optimizer(model, d_model, factor=1, warmup=4000):
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

    def lr_lambda(step):
        step = max(step, 1)
        return factor * (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
