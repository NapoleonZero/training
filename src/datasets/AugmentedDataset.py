import torch
from torch.utils.data import Dataset
from typing import Callable, Iterable, List
from torch import Tensor

class AugmentedDataset(Dataset):
    def __init__(self, dataset: Dataset, transforms: List[Callable]) -> None:
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dataset) * (1 + len(self.transforms))

    def __getitem__(self, idx):
        base_idx = idx // (1 + len(self.transforms))
        aug_idx = idx % (1 + len(self.transforms))
        *x, y = self.dataset[base_idx]

        if aug_idx == 0: 
            return *x, y # Original
        else:
            return self.transforms[aug_idx - 1](*x, y) # Augmented
