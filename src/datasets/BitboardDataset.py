import numpy as np
import pandas as pd
import torch
import math
import glob
import gc
from torch.utils.data import Dataset
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from potatorch.datasets.utils import split_dataset
from datasets.BitboardDecoder import BitboardDecoder

def string_to_matrix(bitboard):
    return np.array([b for b in bitboard], dtype=np.uint8).reshape(8,8).copy()

def string_to_array(bitboard):
    return np.array([b for b in bitboard], dtype=np.uint8)

def uint_to_bits(x, bits = 64):
    return np.unpackbits(np.array([x], dtype='>u8').view(np.uint8))

# TODO: create OversampleDataset wrapper class
class BitboardDataset(Dataset):
    """ Represents a generic Dataset of games
    """
    def __init__(self,
                 dir,
                 filename,
                 from_dump=False,
                 glob=True,
                 preload=True,
                 preload_chunks=True,
                 low_memory=False,
                 fraction=1.0,
                 transform=None,
                 target_transform=None,
                 oversample=False,
                 oversample_factor=2.0,
                 oversample_target=5.0,
                 augment_rate=0.0,
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
        self.augment_rate = augment_rate
        self.seed = seed
        self.debug = debug
        self.low_memory = low_memory
        self.reader = None
        self.path = dir + '/' + filename

        assert not (low_memory and preload)
        assert not (low_memory and from_dump)

        # Load already preprocessed dataset from a numpy binary file
        if self.low_memory:
            self._allocate_decoder()
        elif self.from_dump:
            self.dataset, self.aux, self.scores = self._load_dump(dir, filename)
        else:
            # Load and process dataset leveraging multiprocessing in order to 
            # avoid memory retention: create a pool with only one process
            # and replace it after each task submitted so that the used memory
            # is freed
            with Pool(processes=1, maxtasksperchild=1) as pool:
                # Load dataset from disk
                # result = pool.apply_async(self._join_datasets, (dir, filename,))
                # result = pool.apipe(self._join_datasets, dir, filename)
                # ds = result.get()
                # gc.collect()
                ds = self._join_datasets(dir, filename)
                gc.collect()

                # If preload flag is set, load and preprocess dataset in memory
                if self.preload:
                    # result = pool.apipe(self._preprocess_ds, ds)
                    self.dataset, self.aux, self.scores = self._preprocess_ds(ds)
                    gc.collect()
                    # features, aux, scores = result.get()
                    # self.dataset, self.aux, self.scores = result.get()

                    # del result
                    del ds
                    gc.collect()

        gc.collect()
        if self.debug and not self.low_memory:
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

    def worker_init_fn(self, id, *args):
        self._allocate_decoder()

    def _allocate_decoder(self):
        if self.low_memory:
            self.decoder = BitboardDecoder(self.path, memory_mapped=False)
        
    def _fraction(self, ds):
        """ Return `self.fraction`% of the given dataset """
        # TODO: use this everywhere needed
        return ds[:int(len(ds)*self.fraction)]

    def _load_dump(self, dir, npz):
        """ 
        Load already preprocessed dataset from a numpy binary file
        """
        with np.load(f'{dir}/{npz}') as nps:
            bitboards = nps['bitboards'].copy()
            aux = nps['aux'].copy()
            scores = nps['scores'].copy()
        gc.collect()

        return self._fraction(bitboards), self._fraction(aux), self._fraction(scores)

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

        dtypes = {
                0: 'uint64', 1: 'uint64',   # bitboards
                2: 'uint64', 3: 'uint64',   # bitboards
                4: 'uint64', 5: 'uint64',   # bitboards
                6: 'uint64', 7: 'uint64',   # bitboards
                8: 'uint64', 9: 'uint64',  # bitboards
                10: 'uint64', 11: 'uint64', # bitboards
                12: 'uint8',                # side to move: 0 = white, 1 = black
                13: 'uint8',                # enpassant square: 0-63, 65 is none 
                14: 'uint8',                # castling status: integer value
                15: 'float16'               # score in centipawns: [-inf, +inf]
                }

        dfs = []

        for filename in files:
            if self.debug:
                print(f"Loading {filename}")

            # use dtype=str when parsing the csv to avoid converting bitboards 
            # in integer values (e.g ....001 -> 1)
            df = pd.read_csv(filename, header=0, dtype=dtypes)
            dfs.append(df)

        if self.debug:
            print('Concatenating datasets')

        df_merged = pd.concat(dfs, axis=0, ignore_index=True)
        df_merged = self._filter_scores(df_merged, scaled=False)
        gc.collect()

        
        if self.fraction < 1.0 and self.preload_chunks:
            if self.debug:
                print('Sampling dataset')
            return df_merged.sample(frac=1, random_state=self.seed)[:int(len(df_merged)*self.fraction)]

        return df_merged

    def _filter_scores(self, ds, scaled=False):
        """ Remove positions with too high scores """
        return ds[np.abs(ds['score']) < np.inf]

        # if scaled:
        #     return ds[np.abs(ds[17]) < 2000] 
        # return ds[np.abs(ds[17]) < 2000*100]

    def _preprocess_ds(self, ds):
        """ Preprocess dataset in place.
            Returns: 
                - numpy tensor of 12x8x8 bitboards,
                - numpy array of auxiliary inputs (side, ep, castling)
                - numpy array of scores
        """

        # ds[17] /= 100.0 # we divide by 100 so that score(pawn) = 1
        # change black's score perspective: always evaluate white's position
        # ds.iloc[(ds.iloc[:, 13] == 1).values, 17] *= -1.0

        # ds.drop(ds.columns[[0, 16]], inplace=True, axis=1) # drop fen position and depth
        # ds.reset_index(drop=True, inplace=True)
        # gc.collect()

        if self.debug:
            print(ds.info(memory_usage='deep'))

        aux = ds.iloc[:, -4:-1].values.copy()
        scores = ds.iloc[:, -1].values.copy()
        #######################################################################
        bitboards = ds.iloc[:, :-4].values.copy()
        del ds
        gc.collect()
        #######################################################################

        if self.debug:
            print('Returning (copy of) dataset')
        return bitboards, aux, scores
    

    def _preprocess_row(self, row):
        row = row.copy()
        for i in range(1, 13):
            row[i] = string_to_array(row[i])

        # row[17:] /= 100.0 # we divide by 100 so that score(pawn) = 1

        # change black's score perspective: always evaluate white's position
        if row[13] == 1:
            row[17:] *= -1.0

        row.drop(0, inplace=True) # drop fen position encoding
        row.drop(16, inplace=True) # drop depth
        row.reset_index(drop=True, inplace=True)
        return row

    def __len__(self):
        if self.low_memory:
            return int(self.decoder.length() * self.fraction)

        if self.oversample:
            # return len(self.dataset) + int(len(self.oversample_dataset)*self.oversample_factor)
            return len(self.dataset) + int(len(self.oversample_dataset) * (self.oversample_factor-1))
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.low_memory:
            entry = self.decoder.read_line(idx)
            target = np.array(entry[-1])
            features = np.array(entry[:-4], dtype=np.uint64)
            aux = np.array(entry[-4:-1])
        elif not self.preload:
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

        try:
            # Temporarily treat 'invalid value' warnings as exceptions
            with np.errstate(invalid='raise'):
                features = np.array([uint_to_bits(b).reshape(8,8) for b in features])
        except Exception as e:
            print(f"Exception thrown while decoding: {entry}")
            print(f"features: {features}")
            print(f"Error: {e}")
            exit(1)

        # TODO: test this
        # target = torch.from_numpy(target).float()

        if self.transform:
            (features, aux) = self.transform(features, aux)

        if self.target_transform:
            target = self.target_transform(target)

        features = torch.from_numpy(features).float()
        aux = torch.from_numpy(aux).float()

        if self.augment_rate > 0.0 and np.random.uniform() <= self.augment_rate:
            features = features + torch.randn(12, 8, 8)

        return features, aux, target

    def _drop_oversample_indices(self, indices):
        """ Drop indices that correspond to oversampled items """
        if not self.oversample or not self.oversample_frequency:
            return indices
        
        return [idx for idx in indices if idx % self.oversample_frequency != 0]

    def save_dump(self, file):
        np.savez(file, bitboards=self.dataset, aux=self.aux, scores=self.scores)

