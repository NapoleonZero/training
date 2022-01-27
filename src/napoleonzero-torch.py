#!/usr/bin/python3

import torch
from torch import nn

from datasets import BitboardDataset
from models import CNN
from training import TrainingLoop
import sys

def main():
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    DRIVE_PATH = f'{sys.path[0]}/../datasets'
    DATASET = 'ccrl10M-depth1.csv.part*'

    dataset = BitboardDataset(dir=DRIVE_PATH, filename=DATASET, glob=True, preload=True, fraction=0.02, seed=42, debug=True)
    model = CNN().to(device)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    training_loop = TrainingLoop(
            model,
            dataset,
            loss_fn,
            optimizer,
            train_p=0.7,
            val_p=0.15,
            test_p=0.15,
            batch_size=1024,
            device=device,
            verbose=1
            )

    training_loop.run(epochs=10)

if __name__ == '__main__':
    main()