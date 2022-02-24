import numpy as np
import pandas as pd
import torch
import math
import glob
import gc
from torch.utils.data import Dataset
from multiprocessing import Pool
from datasets.utils import split_dataset

def string_to_matrix(bitboard):
    return np.array([b for b in bitboard], dtype=np.uint8).reshape(8,8).copy()

def string_to_array(bitboard):
    return np.array([b for b in bitboard], dtype=np.uint8)

# TODO: create OversampleDataset wrapper class
class BitboardDataset(Dataset):
    """ Represents a generic Dataset of games
        TODO: implement transform and target_transform
    """
    def __init__(self,
                 dir,
                 filename,
                 from_dump=False,
                 glob=True,
                 preload=True,
                 preload_chunks=True,
                 fraction=1.0,
                 transform=None,
                 target_transform=None,
                 oversample=False,
                 oversample_factor=2.0,
                 oversample_target=5.0,
                 seed=42,
                 debug=False):
        self.dir = dir
        self.from_dump = from_dump
        self.glob = glob
        self.preload = preload
        self.preload_chunks = preload_chunks
        self.fraction = fraction
        self.transform = transform
        self.target_transform = target_transform
        self.oversample = oversample
        self.oversample_factor = oversample_factor
        self.oversample_target = oversample_target
        self.seed = seed
        self.debug = debug

        # Load already preprocessed dataset from a numpy binary file
        if self.from_dump:
            self.dataset, self.aux, self.scores = self._load_dump(dir, filename)
        else:
            # Load and process dataset leveraging multiprocessing in order to 
            # avoid memory retention: create a pool with only one process
            # and replace it after each task submitted so that the used memory
            # is freed
            with Pool(processes=1, maxtasksperchild=1) as pool:
                # Load dataset from disk
                result = pool.apply_async(self._join_datasets, (dir, filename,))
                ds = result.get()
                gc.collect()

                # If preload flag is set, load and preprocess dataset in memory
                if self.preload:
                    result = pool.apply_async(self._preprocess_ds, (ds,))
                    # features, aux, scores = result.get()
                    self.dataset, self.aux, self.scores = result.get()

                    del result
                    del ds
                    gc.collect()

        gc.collect()
        if self.debug:
            print('Dataset initialized')
            print(f'Bitboards size: {self.dataset.nbytes / 2**30}G')
            print(f'Scores size: {self.scores.nbytes / 2**30}G')
            print(f'Aux size: {self.aux.nbytes / 2**30}G')

        if self.oversample:
            assert self.oversample_factor > 1
            self.oversample_factor = int(math.ceil(self.oversample_factor))

            idx = abs(self.scores) > self.oversample_target
            self.oversample_dataset = self.dataset[idx]
            self.oversample_aux = self.aux[idx]
            self.oversample_scores = self.scores[idx]
            self.oversample_frequency = math.ceil(self.__len__() / len(self.oversample_dataset) * (self.oversample_factor - 1))
        

    def _load_dump(self, dir, npz):
        """ 
        Load already preprocessed dataset from a numpy binary file
        """
        with np.load(f'{dir}/{npz}') as nps:
            bitboards = nps['bitboards'].copy()
            aux = nps['aux'].copy()
            scores = nps['scores'].copy()
        gc.collect()

        return bitboards, aux, scores

    def _join_datasets(self, dir, glob_str):
        dfs = []
        name_glob = dir + '/' + glob_str
        if self.glob:
            files = glob.glob(name_glob)
        else:
            files = [name_glob]

        # Sort filenames in ascending order
        files = sorted(files)

        # If preload_chunks = False, we only load the first self.fraction of
        # the dataset chunks. Otherwise we load them all and the return only
        # self.fraction of the whole dataset. By doing so we can perform an
        # unbiased sampling of the dataset when preload_chunks = True.
        if not self.preload_chunks:
            files = files[:int(len(files)*self.fraction)]

        # TODO: try pandas arrow string type
        dtypes = {
                0: 'string',                # fen position string
                1: 'string', 2: 'string',   # bitboards
                3: 'string', 4: 'string',   # bitboards
                5: 'string', 6: 'string',   # bitboards
                7: 'string', 8: 'string',   # bitboards
                9: 'string', 10: 'string',  # bitboards
                11: 'string', 12: 'string', # bitboards
                13: 'uint8',                # side to move: 0 = white, 1 = black
                14: 'uint8',                # enpassant square: 0-63, 65 is none 
                15: 'uint8',                # castling status: integer value
                16: 'uint8',                # depth of search: 1-100
                17: 'float16'               # score in centipawns: [-inf, +inf]
                }

        dfs = []

        for filename in files:
            if self.debug:
                print(f"Loading {filename}")

            # use dtype=str when parsing the csv to avoid converting bitboards 
            # in integer values (e.g ....001 -> 1)
            df = pd.read_csv(filename, header=None, dtype=dtypes)
            dfs.append(df)

        if self.debug:
            print('Concatenating datasets')
        df_merged = pd.concat(dfs, axis=0, ignore_index=True)
        # df_merged = self._filter_scores(df_merged, scaled=False)
        gc.collect()

        
        if self.fraction < 1.0 and self.preload_chunks:
            if self.debug:
                print('Sampling dataset')
            return df_merged.sample(frac=1, random_state=self.seed)[:int(len(df_merged)*self.fraction)]

        return df_merged

    def _filter_scores(self, ds, scaled=False):
        """ Remove positions with too high scores """
        if scaled:
            return ds[np.abs(ds[17]) < 2000] 
        return ds[np.abs(ds[17]) < 2000*100]

    def _preprocess_ds(self, ds):
        """ Preprocess dataset in place.
            Returns: 
                - numpy tensor of 12x8x8 bitboards,
                - numpy array of auxiliary inputs (side, ep, castling)
                - numpy array of scores
        """

        ds[17] /= 100.0 # we divide by 100 so that score(pawn) = 1
        # change black's score perspective: always evaluate white's position
        ds.iloc[(ds.iloc[:, 13] == 1).values, 17] *= -1.0

        ds.drop(ds.columns[[0, 16]], inplace=True, axis=1) # drop fen position and depth
        ds.reset_index(drop=True, inplace=True)
        gc.collect()

        if self.debug:
            print(ds.info(memory_usage='deep'))

        aux = ds.iloc[:, -4:-1].values.copy()
        scores = ds.iloc[:, -1].values.copy()
        #######################################################################
        if self.debug:
            print('Reshaping bitboards')

        bitboards = ds.iloc[:, :-4].values.copy()
        del ds
        gc.collect()

        bitboards = np.array([[string_to_matrix(b) for b in bs] for bs in bitboards])
        #######################################################################

        if self.debug:
            print('Returning (copy of) dataset')
        return bitboards, aux, scores
    

    def _preprocess_row(self, row):
        row = row.copy()
        for i in range(1, 13):
            row[i] = string_to_array(row[i])

        row[17:] /= 100.0 # we divide by 100 so that score(pawn) = 1

        # change black's score perspective: always evaluate white's position
        if row[13] == 1:
            row[17:] *= -1.0

        row.drop(0, inplace=True) # drop fen position encoding
        row.drop(16, inplace=True) # drop depth
        row.reset_index(drop=True, inplace=True)
        return row

    def __len__(self):
        if self.oversample:
            # return len(self.dataset) + int(len(self.oversample_dataset)*self.oversample_factor)
            return len(self.dataset) + int(len(self.oversample_dataset) * (self.oversample_factor-1))
        return len(self.dataset)

    def __getitem__(self, idx):
        if not self.preload:
            entry = self._preprocess_row(self.dataset.iloc[idx])
            target = entry.iloc[-1]
            # Reshape features into array of 12 bitboards of size 8x8
            features = entry[:-4] # drop side, ep, castling, score
            features = np.concatenate(features.values).reshape(12, 8, 8)
            aux = entry[-4:-1] # side, ep, castling
        else: # already preloaded and processed
            if self.oversample and idx % self.oversample_frequency == 0:
                idx = (idx // self.oversample_frequency) % len(self.oversample_dataset)
                features = self.oversample_dataset[idx]
                target = self.oversample_scores[idx]
                aux = self.oversample_aux[idx]
            else:
                idx = idx % len(self.dataset)
                features = self.dataset[idx]
                target = self.scores[idx]
                aux = self.aux[idx]

        return torch.from_numpy(features), torch.from_numpy(aux), target

    # def split(self, train_p=0.7, val_p=0.15, test_p=0.15, seed=None, oversample=False):
    #     """ Split the dataset into disjoint sets: training, validation and test
    #         If the dataset has been oversampled and oversample=True is passed to
    #         this method, the computed validation and test sets will possibly
    #         include the oversampled items.

    #         Note: if oversample=True you won't get a disjoint partition in
    #         general
    #         Note: when using an oversampled dataset and passing
    #         oversample=False, the validation and test datasets will not have
    #         precisely the sizes indicated by the percentages, although
    #         their expected value will.

    #     """
    #     if seed is None:
    #         seed = self.seed

    #     train_ds, val_ds, test_ds = split_dataset(self, train_p, val_p, test_p, seed)

    #     if self.oversample and not oversample:
    #         val_ds.indices = self._drop_oversample_indices(val_ds.indices)
    #         test_ds.indices = self._drop_oversample_indices(test_ds.indices)
    #     return train_ds, val_ds, test_ds

    def _drop_oversample_indices(self, indices):
        """ Drop indices that correspond to oversampled items """
        if not self.oversample or not self.oversample_frequency:
            return indices
        
        return [idx for idx in indices if idx % self.oversample_frequency != 0]

    def save_dump(self, file):
        np.savez(file, bitboards=self.dataset, aux=self.aux, scores=self.scores)

