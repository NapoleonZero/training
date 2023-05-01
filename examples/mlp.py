import random
import numpy as np
import pandas as pd
import math
import torch
import sys
from torch import nn
from functools import partial

from potatorch.training import TrainingLoop
from potatorch.callbacks import TrainingCallback, ProgressbarCallback, LRSchedulerCallback
from potatorch.callbacks import CheckpointCallback, SanityCheckCallback

from torch.utils.data import TensorDataset

device = 'cuda'
SEED = 42

epochs = 100
lr = 1e-4
model = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), 
                      nn.Linear(128, 128), nn.ReLU(),
                      nn.Linear(128, 128), nn.ReLU(),
                      nn.Linear(128, 128), nn.ReLU(),
                      nn.Linear(128, 1)).to(device)
dataset = TensorDataset(torch.arange(1000).view(1000, 1), torch.sin(torch.arange(1000)))
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

training_loop = TrainingLoop(
    model,
    dataset,
    loss_fn,
    optimizer,
    train_p=0.8,
    val_p=0.1,
    test_p=0.1,
    batch_size=256,
    shuffle=True,
    device=device,
    verbose=1,
    seed=SEED,
    val_metrics={'l1': nn.L1Loss(), 'mse': nn.MSELoss()},
    callbacks=[
        ProgressbarCallback(epochs=epochs, width=20),
    ]
)
model = training_loop.run(epochs=epochs)
