import wandb
import torch
import os

def download_wandb_checkpoint(run_path, filename, device='cuda'):
    api = wandb.Api()

    run = api.run(run_path)
    run.file(filename).download(replace=True)
    checkpoint = torch.load(filename, map_location=torch.device(device))
    return checkpoint

def save_wandb_file(path):
    wandb.save(path, base_path=os.path.dirname(path))
