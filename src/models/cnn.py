import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_stack = nn.Sequential(
                nn.Conv2d(12, 32, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(2,2), stride=1, padding='same'),
                nn.ReLU()
                )

        self.linear_output_stack = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(128),
                nn.ReLU(),
                nn.LazyLinear(128),
                nn.ReLU(),
                nn.LazyLinear(1)
                )

    def forward(self, x):
        x = self.conv_stack(x)
        y = self.linear_output_stack(x)
        return y
