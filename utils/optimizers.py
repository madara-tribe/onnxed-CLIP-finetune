import torch

def create_optimizers(model, cfg, num_train):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_train*cfg.epochs)
    return optimizer, scheduler

