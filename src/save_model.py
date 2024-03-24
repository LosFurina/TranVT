import src.models
from src.constant import Args

import os
import torch


def save_model(model, optimizer, scheduler, accuracy_list, args: Args):
    file_path = os.path.join(args.exp_path, "checkpoint_model.ckpt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)
    args.logger.info("Save checkpoint at {}".format(file_path))


def load_model(load_path: str, args: Args):
    model = src.models.TranVT(0, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0, gamma=0)

    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    accuracy_list = checkpoint['accuracy_list']
    return model, optimizer, scheduler, accuracy_list
