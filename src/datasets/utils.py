import torch
from torch.utils.data import random_split


def split_dataset(ds, train_p=0.7, val_p=0.15, test_p=0.15, seed=42):
    """ Split a torch.utils.data.Datset into 3 random split, according to the
        given percentages.

        train_p, val_p, test_p must sum up to 1

        returns (train_ds, val_ds, test_ds)
    """
    train_size = int(len(ds) * train_p)

    # if ds.oversample:
    #     val_size = int(len(ds) * val_p)
    #     test_size = len(ds) - train_size - val_size
    # else:
    val_size = int(len(ds) * val_p)
    test_size = len(ds) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(ds,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
            )
    # for clarity
    return train_ds, val_ds, test_ds
