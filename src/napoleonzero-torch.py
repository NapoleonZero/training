#!/usr/bin/python3

import random
import numpy as np
import torch
import math
from torch import nn

from datasets import BitboardDataset, FilteredDataset
from models import CNN, BitboardTransformer
from utils import download_wandb_checkpoint
from training import TrainingLoop
from callbacks import TrainingCallback, ProgressbarCallback, LRSchedulerCallback
from callbacks import WandbCallback, CheckpointCallback
from datasets.BitboardDataset import string_to_matrix
import sys

def set_random_state(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_summary(model):
    print(model)
    print(f'Number of parameters: {params_count(model)}')

def read_bitboards(csv):
  """ csv: comma separated set of bitboards.
           Each element of the set is a string of 64 binary values.

      returns: np.array of shape 12x8x8
  """
  bitboards = csv.split(',')
  return np.array([string_to_matrix(b) for b in bitboards])

def main():
    SEED = 42
    set_random_state(SEED)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    DRIVE_PATH = f'{sys.path[0]}/../datasets'
    ARTIFACTS_PATH = f'{sys.path[0]}/../artifacts'
    DATASET = 'positions-12M.npz'

    # dataset = BitboardDataset(dir=DRIVE_PATH, filename=DATASET, glob=False, preload=True, preload_chunks=True, fraction=1.0, seed=SEED, debug=True)
    oversample = False
    oversample_factor=None
    oversample_target=None
    filter_threshold=200
    dataset = BitboardDataset(
                dir=DRIVE_PATH, filename=DATASET, seed=SEED,
                preload=True,
                from_dump=True,
                oversample=oversample,
                oversample_factor=oversample_factor,
                oversample_target=oversample_target,
                debug=True
                )
    print('Filtering dataset')
    # x[0] = bitboards, x[1] = aux, x[2] = score
    dataset = FilteredDataset(dataset, lambda x: abs(x[2]) < filter_threshold)
    print('Done filtering')

    patch_size = 1
    # TODO: retrieve some of this stuff automatically from TrainingLoop during callback 
    # TODO: move to a YAML file
    config = {
            'seed': SEED,
            'device': device,
            'dataset': DATASET,
            'dataset-size': len(dataset),
            'oversample': oversample,
            'oversample_factor': oversample_factor,
            'oversample_target': oversample_target,
            'filter-threshold': filter_threshold,
            'cnn-projection': True,
            'cnn-output-channels': 128,
            'cnn-layers': 4,
            'cnn-kernel-size': 3,
            'cnn-residual': True,
            'cnn-pool': True,
            'vit-patch-size': patch_size,
            'vit-dim': 128,
            'vit-depth': 6,
            'vit-heads': 16,
            'vit-mlp-dim': 256,
            'vit-dropout': 0.1,
            'vit-emb-dropout': 0.0,
            'material_head': False,
            'weight-decay': 0.0,
            'learning-rate': 1e-3,
            'lr-warmup-steps': 1000,
            'lr-cosine-annealing': True,
            'lr-cosine-tmax': 200,
            'lr-cosine-factor': 1,
            'lr-restart': False,
            'min-lr': 1e-6,
            'adam-betas': (0.9, 0.999),
            'epochs': 200,
            'train-split-perc': 0.975,
            'val-split-perc': 0.0125,
            'test-split-perc': 0.0125,
            'batch-size': 2**12,
            'shuffle': True,
            'mixed-precision': True,
            }

    model = BitboardTransformer(
                cnn_projection=config['cnn-projection'],
                cnn_out_channels=config['cnn-output-channels'],
                cnn_layers=config['cnn-layers'],
                cnn_kernel_size=config['cnn-kernel-size'],
                cnn_residual=config['cnn-residual'],
                cnn_pool=config['cnn-pool'],
                patch_size=config['vit-patch-size'],
                dim=config['vit-dim'],
                depth=config['vit-depth'],
                heads=config['vit-heads'],
                mlp_dim=config['vit-mlp-dim'],
                material_head=config['material_head'],
                dropout=config['vit-dropout'],
                emb_dropout=config['vit-emb-dropout']
            ).to(device, memory_format=torch.channels_last)
    print_summary(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.RAdam(
            model.parameters(),
            betas=config['adam-betas'],
            weight_decay=config['weight-decay'],
            lr=config['learning-rate']
            )

    epochs = config['epochs']

    #TODO: maybe pass config as parameter
    training_loop = TrainingLoop(
            model,
            dataset,
            loss_fn,
            optimizer,
            train_p=config['train-split-perc'],
            val_p=config['val-split-perc'],
            test_p=config['test-split-perc'],
            batch_size=config['batch-size'],
            shuffle=config['shuffle'],
            device=device,
            mixed_precision=config['mixed-precision'],
            verbose=1,
            seed=SEED,
            val_metrics={'l1': nn.L1Loss()},
            callbacks=[
                LRSchedulerCallback(
                    optimizer,
                    warmup_steps=config['lr-warmup-steps'],
                    cosine_annealing=config['lr-cosine-annealing'],
                    cosine_tmax=config['lr-cosine-tmax'],
                    cosine_factor=config['lr-cosine-factor'],
                    restart=config['lr-restart'],
                    min_lr=config['min-lr']
                    ),
                ProgressbarCallback(
                    epochs=epochs,
                    width=20),
                WandbCallback(
                    project_name='napoleon-zero-pytorch',
                    entity='marco-pampaloni',
                    config=config,
                    tags=['test-new-dataset-12M']
                    ),
                CheckpointCallback(
                    path=ARTIFACTS_PATH + '/checkpoint.pt',
                    save_best=True,
                    metric='val_loss',
                    sync_wandb=True,
                    debug=True
                    )
                ]
            )
    training_loop.run(epochs=epochs)

    # checkpoint = download_wandb_checkpoint('marco-pampaloni/napoleon-zero-pytorch/36q9akms', 'checkpoint.pt')
    # training_loop.load_state(checkpoint)
    # training_loop.run(epochs=epochs)
if __name__ == '__main__':
    main()