#!/usr/bin/python3

import random
import numpy as np
import pandas as pd
import math
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch._C import MobileOptimizerType
from functools import partial

from datasets import BitboardDataset, FilteredDataset
from models import CNN, BitboardTransformer 
from utils import download_wandb_checkpoint, save_wandb_file
from training import TrainingLoop
from callbacks import TrainingCallback, ProgressbarCallback, LRSchedulerCallback
from callbacks import WandbCallback, CheckpointCallback, SanityCheckCallback
from datasets.BitboardDataset import string_to_matrix
from losses import MSRELoss, WDLLoss
import sys

def set_random_state(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_summary(model, print_model=False):
    if print_model:
        print(model)
        print(f'Number of parameters: {params_count(model)}')

def read_bitboards(csv):
  """ csv: comma separated set of bitboards.
           Each element of the set is a string of 64 binary values.

      returns: np.array of shape 12x8x8
  """
  bitboards = csv.split(',')
  return np.array([string_to_matrix(b) for b in bitboards])

def read_positions(file):
    """ Read positions from file and returns the following information:
        - descriptors (either fen of description string) list
        - positions (bitboards) tensor
        - auxiliary inputs (side, ep square, castling) tensor
    """
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
            15: 'uint8'                 # castling status: integer value
            }
    df = pd.read_csv(file, header=None, dtype=dtypes)

    fens = df.iloc[:, 0].values
    bitboards = df.iloc[:, 1:13].values
    aux = df.iloc[:, 13:].values

    x = [(np.array([[string_to_matrix(b) for b in bs]])) for bs in bitboards]
    aux = [(np.array([v])) for v in aux]

    return fens, x, aux

scales = [0.1, 0.3, 0.3, 0.5, 0.9, 1.0]

def rescale_bitboards(bs):
    """ Scale each bitboard by [0.1, 0.2, ..., 0.6] relatively for both colors """
    # return np.array([bs[i] * (i%6 + 1) / 10 for i in np.arange(12)])
    return np.array([bs[i] * scales[i % 6] * 100 * ((i // 6) * (-2) + 1) for i in np.arange(12)])

def rescale_aux(aux):
    return np.array([aux[-3], aux[-2] / 65, aux[-1] / 15])

def filter_scores(filter_threshold: int, data: tuple) -> bool:
    """ Filter scores based on a given cutoff threshold """
    return abs(data[2]) < filter_threshold

def main():
    SEED = 42
    torch.backends.cuda.matmul.allow_tf32 = True
    set_random_state(SEED)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    DRIVE_PATH = f'{sys.path[0]}/../datasets'
    ARTIFACTS_PATH = f'{sys.path[0]}/../artifacts'
    DATASET = 'lichess151M-preprocessed.bin'

    (check_fen, check_x, check_aux) = read_positions(f'{sys.path[0]}/sanity_check.csv')

    oversample = False
    oversample_factor = None
    oversample_target = None
    target_scale = 1e-3
    filter_threshold = 2000 * target_scale
    mate_value = filter_threshold * 0.9999
    augment_rate = 0.0
    dataset = BitboardDataset(
                dir=DRIVE_PATH, filename=DATASET, seed=SEED,
                glob=False,
                preload=False,
                from_dump=False,
                low_memory=True,
                oversample=oversample,
                oversample_factor=oversample_factor,
                oversample_target=oversample_target,
                fraction=1.0,
                # transform=(lambda x, aux: (rescale_bitboards(x), aux)),
                transform=(lambda x, aux: (rescale_bitboards(x), rescale_aux(aux))),
                target_transform=(lambda y: y * 100 * target_scale), # for stockfish evaluations
                augment_rate=augment_rate,
                debug=True
                )

    base_lr = 1e-3
    batch_size = 2**10
    patch_size = 2
    epochs = 200
    # TODO: retrieve some of this stuff automatically from TrainingLoop during callback 
    # TODO: move to a YAML file
    # TODO: multiplying bitboards planes seems to be helpful (e.g. pawn * 0.1, bishop * 0.2, etc.)
    config = {
            'seed': SEED,
            'device': device,
            'dataset': DATASET,
            'dataset-size': len(dataset),
            'oversample': oversample,
            'oversample_factor': oversample_factor,
            'oversample_target': oversample_target,
            'filter-threshold': filter_threshold,
            'augment-rate': augment_rate,
            'cnn-projection': True,
            'cnn-resnext': True,
            'cnn-output-channels': 256,
            'cnn-layers': 6,
            'cnn-kernel-size': 3,
            'cnn-residual': True,
            'cnn-pool': False,
            'cnn-depthwise': False,
            'cnn-squeeze': True,
            'vit-patch-size': patch_size,
            'vit-dim': 512,
            'vit-heads': 8,
            'vit-stages-depth': [4],
            'vit-merging-strategy': '1d', # TODO: test 2d mode more throughly
            'vit-mlp-dim': 256,
            'vit-dropout': 0.01,
            'vit-emb-dropout': 0.0,
            'vit-stochastic-depth-p': 0.0,
            'vit-stochastic-depth-mode': 'row',
            'vit-random-patch-projection': False,
            'vit-channel-pos-encoding': True,
            'vit-learned-pos-encoding': False,
            'material_head': False,
            'weight-decay': 1e-3,
            'learning-rate': base_lr * batch_size / 256,
            'lr-warmup-steps': int(10000 / (batch_size / 2**10)),
            'lr-cosine-annealing': True,
            'lr-cosine-tmax': epochs,
            'lr-cosine-factor': 1,
            'lr-restart': False,
            'min-lr': 1e-6,
            'adam-betas': (0.95, 0.99),
            'epochs': epochs,
            'train-split-perc': 0.995,
            'val-split-perc': 0.0025,
            'test-split-perc': 0.0025,
            'batch-size': batch_size,
            'shuffle': False,
            'random-subsampling': 0.001,
            'mixed-precision': True,
            }

    model = BitboardTransformer(
                channel_aux_encoding=True,
                cnn_projection=config['cnn-projection'],
                cnn_resnext=config['cnn-resnext'],
                cnn_out_channels=config['cnn-output-channels'],
                cnn_layers=config['cnn-layers'],
                cnn_kernel_size=config['cnn-kernel-size'],
                cnn_residual=config['cnn-residual'],
                cnn_pool=config['cnn-pool'],
                cnn_depthwise=config['cnn-depthwise'],
                cnn_squeeze=config['cnn-squeeze'],
                stages_depth=config['vit-stages-depth'],
                merging_strategy=config['vit-merging-strategy'],
                stochastic_depth_p=config['vit-stochastic-depth-p'],
                stochastic_depth_mode=config['vit-stochastic-depth-mode'],
                patch_size=config['vit-patch-size'],
                dim=config['vit-dim'],
                heads=config['vit-heads'],
                mlp_dim=config['vit-mlp-dim'],
                random_patch_projection=config['vit-random-patch-projection'],
                channel_pos_encoding=config['vit-channel-pos-encoding'],
                learned_pos_encoding=config['vit-learned-pos-encoding'],
                material_head=config['material_head'],
                dropout=config['vit-dropout'],
                emb_dropout=config['vit-emb-dropout']
            ).to(device, memory_format=torch.channels_last)
    print_summary(model)

    # loss_fn = nn.MSELoss()
    log_loss = MSRELoss(delta=(filter_threshold + 1))
    loss_fn = WDLLoss(input_scale=300*target_scale, target_scale=300*target_scale)
    # TODO: try gradient clipping
    optimizer = torch.optim.AdamW(
            [{'params': model.cnn.parameters()},
                {'params': model.vit.parameters(), 'lr': 1e-5}
                ],
            # model.parameters(),
            betas=config['adam-betas'],
            weight_decay=config['weight-decay'],
            lr=config['learning-rate']
            )

    epochs = config['epochs']

    sanity_check_callback = SanityCheckCallback(
            data=list(zip(check_x, check_aux)),
            descriptors=check_fen,
            # transform=(lambda x, aux: (rescale_bitboards(x), aux)),
            transform=(lambda x, aux: (x, rescale_aux(aux))),
            target_transform=(lambda y: y / target_scale)
            )
    wandb_callback = WandbCallback(
            project_name='napoleon-zero-pytorch',
            entity='marco-pampaloni',
            config=config,
            tags=['test-new-dataset-12M']
            )
    checkpoint_callback = CheckpointCallback(
            path=ARTIFACTS_PATH + '/checkpoint.pt',
            save_best=True,
            metric='val_loss',
            sync_wandb=True,
            debug=True
            )
    anomaly_detection = CheckpointCallback(
            path=ARTIFACTS_PATH + '/anomaly_checkpoint.pt',
            save_best=False,
            detect_anomaly=True,
            metric='mean_loss',
            sync_wandb=True,
            debug=True
            )

    #TODO: maybe pass config as parameter
    training_loop = TrainingLoop(
            model,
            dataset,
            loss_fn,
            optimizer,
            train_p=config['train-split-perc'],
            val_p=config['val-split-perc'],
            test_p=config['test-split-perc'],
            batch_size=config['batch-size'],
            shuffle=config['shuffle'],
            random_subsampling=config['random-subsampling'],
            filter_fn=partial(filter_scores, filter_threshold),
            device=device,
            mixed_precision=config['mixed-precision'],
            verbose=1,
            seed=SEED,
            val_metrics={'l1': nn.L1Loss(), 'mse': nn.MSELoss(), 'logl': log_loss},
            callbacks=[
                LRSchedulerCallback(
                    optimizer,
                    warmup_steps=config['lr-warmup-steps'],
                    cosine_annealing=config['lr-cosine-annealing'],
                    cosine_tmax=config['lr-cosine-tmax'],
                    cosine_factor=config['lr-cosine-factor'],
                    restart=config['lr-restart'],
                    min_lr=config['min-lr']
                    ),
                ProgressbarCallback(
                    epochs=epochs,
                    width=20),
                # wandb_callback,
                # checkpoint_callback,
                # anomaly_detection,
                sanity_check_callback
                ]
            )
    model = training_loop.run(epochs=epochs)

    # torch.autograd.set_detect_anomaly(True)
    # checkpoint = download_wandb_checkpoint('marco-pampaloni/napoleon-zero-pytorch/1pkd6poq', 'anomaly_checkpoint.pt', device=device)
    # training_loop.load_state(checkpoint)
    # model = training_loop.run(epochs=epochs)

    # wandb_callback.init()
    # serialize(model, training_loop, f'{ARTIFACTS_PATH}/script.pt', optimize=False)

def serialize(model, training_loop, path, optimize=True):
    # torch._C._jit_set_profiling_executor(False)
    model.eval()

    serialized_model = torch.jit.script(model)
    if optimize:
        serialized_model = optimize_for_mobile(
                serialized_model,
                optimization_blocklist=set([MobileOptimizerType.INSERT_FOLD_PREPACK_OPS, MobileOptimizerType.HOIST_CONV_PACKED_PARAMS]),
                backend='CPU')

    serialized_model = torch.jit.freeze(serialized_model)
    serialized_model = torch.jit.optimize_for_inference(serialized_model)
    serialized_model.save(path)
    save_wandb_file(path)
    return serialized_model


def evaluate(sanity_check_callback, training_loop):
    print(f'Test dataset evaluation: {training_loop.evaluate()}')
    print('Sanity check positions evaluation:')
    sanity_check_callback.on_train_epoch_end(training_loop)

if __name__ == '__main__':
    main()