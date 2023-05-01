import torch
from torch import nn

from potatorch.training import TrainingLoop
from potatorch.callbacks import ProgressbarCallback
from torch.utils.data import TensorDataset

# Fix a seed for TrainingLoop to make non-deterministic operations such as
# shuffling reproducible
SEED = 42
device = 'cuda'

epochs = 100
lr = 1e-4

# Define your model as a pytorch Module
model = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), 
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1))

# Create your dataset as a torch.data.Dataset
dataset = TensorDataset(torch.arange(1000).view(1000, 1), torch.sin(torch.arange(1000)))

# Provide a loss function and an optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Construct a TrainingLoop object.
# TrainingLoop handles the initialization of dataloaders, dataset splitting,
# shuffling, mixed precision training, etc.
# You can provide callback handles through the `callbacks` argument.
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
# Run the training loop
model = training_loop.run(epochs=epochs)
