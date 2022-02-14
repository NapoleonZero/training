import wandb
import torch

def download_wandb_checkpoint(run_path, filename):
    api = wandb.Api()

    run = api.run(run_path)
    run.file(filename).download(replace=True)
    checkpoint = torch.load(filename)
    return checkpoint
