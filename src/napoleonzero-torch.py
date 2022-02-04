#!/usr/bin/python3

import random
import numpy as np
import torch
from torch import nn

from datasets import BitboardDataset
from models import CNN, BitboardTransformer
from training import TrainingLoop
from callbacks import TrainingCallback, ProgressbarCallback, LRSchedulerCallback
import sys

def set_random_state(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    SEED = 42
    set_random_state(SEED)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    DRIVE_PATH = f'{sys.path[0]}/../datasets'
    DATASET = 'ccrl10M-depth1.csv.part*'

    dataset = BitboardDataset(dir=DRIVE_PATH, filename=DATASET, glob=True, preload=True, preload_chunks=False, fraction=0.02, seed=SEED, debug=True)

    patch_size = 4
    model = BitboardTransformer(
                patch_size=patch_size,
                dim=(patch_size**2 * 8),
                depth=8,
                heads=16,
                mlp_dim=256,
                dropout=0.10,
                emb_dropout=0.0
            ).to(device, memory_format=torch.channels_last)
    print(model)
    print(f'Number of parameters: {params_count(model)}')


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.RAdam(model.parameters(), betas=(0.9, 0.999), weight_decay=0.0, lr=1e-3)
    epochs = 200

    training_loop = TrainingLoop(
            model,
            dataset,
            loss_fn,
            optimizer,
            train_p=0.95,
            val_p=0.025,
            test_p=0.025,
            batch_size=2**8,
            shuffle=True,
            device=device,
            mixed_precision=True,
            verbose=1,
            seed=SEED,
            callbacks=[
                LRSchedulerCallback(optimizer, warmup_steps=1000),
                ProgressbarCallback(epochs=epochs, width=20)
                ]
            )

    training_loop.run(epochs=epochs)

if __name__ == '__main__':
    main()