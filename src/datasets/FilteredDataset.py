import torch
from torch.utils.data import Dataset

class FilteredDataset(Dataset):
    """ Wrapper class that filters a dataset with a provided filtering function.

        - `filter_fn` is such that:
           new_dataset = [d for d in old_dataset if filter_fn(d)]
    """
    def __init__(self, dataset, filter_fn):
        self.dataset = dataset
        self.indices = [idx for idx in range(len(dataset)) if filter_fn(dataset[idx])]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
