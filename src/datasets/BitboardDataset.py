import numpy as np
import pandas as pd
import torch
import glob
import gc
from torch.utils.data import Dataset

def string_to_matrix(bitboard):
    return np.array([b for b in bitboard], dtype=np.uint8).reshape(8,8).copy()

def string_to_array(bitboard):
    return np.array([b for b in bitboard], dtype=np.uint8)

class BitboardDataset(Dataset):
    """ Represents a generic Dataset of games
        TODO: implement transform and target_transform
    """
    def __init__(self, dir, filename, glob=True, preload=True, fraction=1.0, transform=None, target_transform=None, seed=42, debug=False):
        self.dir = dir
        self.glob = glob
        self.preload = preload
        self.fraction = fraction
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed
        self.debug = debug

        # Load dataset from disk
        self.dataset = self._join_datasets(dir, filename)
        # If preload flag is set, load and preprocess dataset in memory
        if self.preload:
            features, self.aux, self.scores = self._preprocess_ds(self.dataset)
            del self.dataset
            gc.collect()
            self.dataset = features

        gc.collect()


    def _join_datasets(self, dir, glob_str):
        dfs = []
        name_glob = dir + '/' + glob_str
        if self.glob:
            files = glob.glob(name_glob)
        else:
            files = [name_glob]

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

        df_merged = None

        for filename in sorted(files):
            if self.debug:
                print(f"Loading {filename}")

            # use dtype=str when parsing the csv to avoid converting bitboards 
            # in integer values (e.g ....001 -> 1)
            df = pd.read_csv(filename, header=None, dtype=dtypes)

            if df_merged is None:
                df_merged = df
            else:
                df_merged = pd.concat([df_merged, df], axis=0, ignore_index=True)
                del df
                gc.collect()

        gc.collect()
        df_merged = self._filter_scores(df_merged, scaled=False)

        if self.debug:
            print('Concatenating datasets')

        if self.fraction < 1.0:
            return df_merged.sample(frac=1, random_state=self.seed)[:int(len(df_merged)*self.fraction)]
        return df_merged

    def _filter_scores(self, ds, scaled=False):
        """ Remove positions with too high scores """
        if scaled:
            return ds[np.abs(ds[17]) < 50] 
        return ds[np.abs(ds[17]) < 50*100]

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

        for i in range(1, 13):
            if self.debug:
                print(f"Processing bitboards for column {i}/12")
                ds[i] = ds[i].apply(string_to_matrix)
                gc.collect()

        ds.drop(ds.columns[[0, 16]], inplace=True, axis=1) # drop fen position and depth
        ds.reset_index(drop=True, inplace=True)
        gc.collect()

        if self.debug:
            print(self.dataset.info(memory_usage='deep'))

        np_dataset = ds.iloc[:, :-4].copy().values
        np_dataset = np.array([np.concatenate(bs).reshape(12, 8, 8).copy() for bs in np_dataset])
        aux = ds.iloc[:, -4:-1].copy().values
        scores = ds.iloc[:, -1].copy().values
        gc.collect()

        return np_dataset, aux, scores

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
        return len(self.dataset)

    def __getitem__(self, idx):
        if not self.preload:
            entry = self._preprocess_row(self.dataset.iloc[idx])
            target = entry.iloc[-1]
            # Reshape features into array of 12 bitboards of size 8x8
            features = entry[:-4] # drop side, ep, castling, score (TODO: find a way to use them)
            features = np.concatenate(features.values).reshape(12, 8, 8)
            aux = entry[-4:-1] # side, ep, castling
        else: # already preloaded and processed
            features = self.dataset[idx]
            target = self.scores[idx]
            aux = self.aux[idx]

        return torch.from_numpy(features), torch.from_numpy(aux), target
