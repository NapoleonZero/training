#!/usr/bin/python3

import random
import numpy as np
import torch
from torch import nn

from datasets import BitboardDataset
from models import CNN
from training import TrainingLoop
import sys

def set_random_state(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def main():
    SEED = 42
    set_random_state(SEED)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    DRIVE_PATH = f'{sys.path[0]}/../datasets'
    DATASET = 'ccrl10M-depth1.csv.part*'

    dataset = BitboardDataset(dir=DRIVE_PATH, filename=DATASET, glob=True, preload=True, fraction=0.05, seed=SEED, debug=True)
    model = CNN().to(device)
    print(model)

    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.7, weight_decay=1e-3, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    training_loop = TrainingLoop(
            model,
            dataset,
            loss_fn,
            optimizer,
            train_p=0.7,
            val_p=0.15,
            test_p=0.15,
            batch_size=2**12,
            shuffle=True,
            device=device,
            mixed_precision=True,
            verbose=1,
            seed=SEED
            )

    training_loop.run(epochs=100)

if __name__ == '__main__':
    main()