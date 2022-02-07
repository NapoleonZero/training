#!/usr/bin/python3

import random
import numpy as np
import torch
from torch import nn

from datasets import BitboardDataset
from models import CNN, BitboardTransformer
from training import TrainingLoop
from callbacks import TrainingCallback, ProgressbarCallback, LRSchedulerCallback
from callbacks import WandbCallback
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

def main():
    SEED = 42
    set_random_state(SEED)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    DRIVE_PATH = f'{sys.path[0]}/../datasets'
    DATASET = 'ccrl10M-depth1.csv.part*'

    dataset = BitboardDataset(dir=DRIVE_PATH, filename=DATASET, glob=True, preload=True, preload_chunks=True, fraction=1.0, seed=SEED, debug=True)

    patch_size = 4
    # TODO: retrieve some of this stuff automatically from TrainingLoop during callback 
    config = {
            'seed': SEED,
            'device': device,
            'dataset': DATASET,
            'dataset-size': len(dataset),
            'vit-patch-size': patch_size,
            'vit-dim': (patch_size**2 * 8),
            'vit-depth': 8,
            'vit-heads': 16,
            'vit-mlp-dim': 256,
            'vit-dropout': 0.2,
            'vit-emb-dropout': 0.0,
            'weight-decay': 0.0,
            'learning-rate': 1e-3,
            'lr-warmup-steps': 1000,
            'lr-cosine-annealing': True,
            'lr-cosine-tmax': 50,
            'lr-restart': False,
            'min-lr': 1e-6,
            'adam-betas': (0.9, 0.999),
            'epochs': 200,
            'train-split-perc': 0.95,
            'val-split-perc': 0.025,
            'test-split-perc': 0.025,
            'batch-size': 2**12,
            'shuffle': True,
            'mixed-precision': True,
            }

    model = BitboardTransformer(
                patch_size=config['vit-patch-size'],
                dim=config['vit-dim'],
                depth=config['vit-depth'],
                heads=config['vit-heads'],
                mlp_dim=config['vit-mlp-dim'],
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
            callbacks=[
                LRSchedulerCallback(
                    optimizer,
                    warmup_steps=config['lr-warmup-steps'],
                    cosine_annealing=config['lr-cosine-annealing'],
                    cosine_tmax=config['lr-cosine-tmax'],
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
                    tags=['initial-test-wandb', 'test-annealing']
                    )
                ]
            )

    training_loop.run(epochs=epochs)

if __name__ == '__main__':
    main()