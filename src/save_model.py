import logging

import yaml

import src.models
from src.constant import Args

import os
import torch


def save_model(model, optimizer, scheduler, accuracy_list, args: Args, logger: logging.Logger):
    file_path = os.path.join(args.exp_path, "checkpoint_model.ckpt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)
    logger.info("Save checkpoint at {}".format(file_path))


def load_model(exp_id: str, save_pattern: str, dim: int, args: Args):
    checkpoint_path = os.path.join("exp", save_pattern, str(exp_id), "checkpoint_model.ckpt")
    model = None
    if args.model == "TranVTV":
        model = src.models.TranVTV(dim, args)
    elif args.model == "TranVTP":
        model = src.models.TranVTP(dim, args)
    elif args.model == "TranVTS":
        model = src.models.TranVTS(dim, args)
    elif args.model == "GTranVTV":
        model = src.models.GTranVTV(dim, args)
    else:
        raise ValueError("Model type not supported")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0, gamma=0)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    accuracy_list = checkpoint['accuracy_list']
    return model, optimizer, scheduler, accuracy_list
