#!/usr/bin/python3

import yaml
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from datasets import BitboardDataset, FilteredDataset
from models import CNN, BitboardTransformer 
from potatorch.utils import download_wandb_checkpoint, save_wandb_file
from potatorch.training import TrainingLoop
from potatorch.callbacks import TrainingCallback, ProgressbarCallback, LRSchedulerCallback
from potatorch.callbacks import WandbCallback, CheckpointCallback, SanityCheckCallback
from datasets.BitboardDataset import string_to_matrix
from losses import MSRELoss
import sys

def set_random_state(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_summary(model):
    print(model)
    print(f'Number of parameters: {params_count(model)}')


# TODO: move these utility functions to a separate file
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

def rescale_bitboards(bs):
    """ Scale each bitboard by [0.1, 0.2, ..., 0.6] relatively for both colors """
    return np.array([bs[i] * (i%6 + 1) / 10 for i in np.arange(12)])

def filter_scores(filter_threshold: int, data: tuple) -> bool:
    """ Filter scores based on a given cutoff threshold """
    return abs(data[2]) < filter_threshold

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def make_loss_fn(name: str) -> nn.Module:
    if name == 'mse':
        return nn.MSELoss()
    raise Exception(f"{name}: Loss not implemented yet.")

class CustomTrainingLoop(TrainingLoop):
    def forward(self, inputs, *args, **kwargs) -> Tensor:
        (X, aux, ys) = inputs
        X = X.float()
        aux = aux.float()
        ys = ys.float()

        pred = self.model(X, aux)
        return pred

    def compute_loss(self, inputs, pred, *args, **kwargs) -> Tensor:
        (X, aux, ys) = inputs
        ys = ys.float()
        loss = self.loss_fn(pred.view(-1), ys)
        return loss

def main():
    config = load_config(f'{sys.path[0]}/../config/vit_training.yaml')

    SEED = config['general']['seed']
    DRIVE_PATH = f'{sys.path[0]}/../datasets/datasets'
    ARTIFACTS_PATH = f'{sys.path[0]}/../artifacts'

    set_random_state(SEED)

    # Get cpu or gpu device for training.
    device = config['general']['device']
    print(f"Using {device} device")

    torch.backends.cuda.matmul.allow_tf32 = True # TODO: why?

    (check_fen, check_x, check_aux) = read_positions(f'{sys.path[0]}/sanity_check.csv')
    check_y = np.zeros((len(check_x), 1)) # mock target (not used, but necessary)

    # filter_threshold = 2000 * target_scale
    # mate_value = filter_threshold * 0.9999
    target_scale = config['general']['target_scale'] # further rescale the target score so that it's a smaller value
    dataset = BitboardDataset(
                dir=DRIVE_PATH,
                seed=SEED,
                glob=False,
                preload=False,
                from_dump=False,
                low_memory=True,
                fraction=1.0,
                target_transform=(lambda y: y * 100 * target_scale), # scores are in centipawns / 100
                debug=True,
                **config['dataset'],
                # transform=(lambda x, aux: (rescale_bitboards(x), aux)),
                )
    config['dataset']['size'] = len(dataset)

    # TODO: retrieve some of this stuff automatically from TrainingLoop during callback (what?)
    # TODO: multiplying bitboards planes seems to be helpful (e.g. pawn * 0.1, bishop * 0.2, etc.)

    model = BitboardTransformer(**config['model'])
    model = model.to(device, memory_format=torch.channels_last) # linter error for some reason
    print_summary(model)

    # TODO: try gradient clipping
    optimizer = torch.optim.AdamW(
            model.parameters(),
            **{**config['optimizer'], 'betas': tuple(config.get('optimizer', {}).get('betas', [0.9, 0.999]))}
            )

    sanity_check_callback = SanityCheckCallback(
            data=list(zip(check_x, check_aux, check_y)),
            descriptors=check_fen,
            # transform=(lambda x, aux: (rescale_bitboards(x), aux)),
            target_transform=(lambda y: y / target_scale)
            )
    wandb_callback = WandbCallback(
            project_name='napoleon-zero-pytorch',
            entity='marco-pampaloni',
            config=config,
            tags=config['general']['tags']
            )
    checkpoint_callback = CheckpointCallback(
            path=ARTIFACTS_PATH + '/checkpoint.pt',
            save_best=True,
            metric='val_loss',
            sync_wandb=True,
            debug=True
            )
    # anomaly_detection = CheckpointCallback(
    #         path=ARTIFACTS_PATH + '/anomaly_checkpoint.pt',
    #         save_best=False,
    #         detect_anomaly=True,
    #         metric='mean_loss',
    #         sync_wandb=True,
    #         debug=True
    #         )

    l1 = lambda pred, inputs: F.l1_loss(pred.view(-1), inputs[-1])
    mse = lambda pred, inputs: F.mse_loss(pred.view(-1), inputs[-1])

    # loss_fn = MSRELoss(delta=(filter_threshold + 1))
    epochs = config['general']['epochs']
    loss_fn = make_loss_fn(config['general']['loss_function'])
    lr_scheduler = LRSchedulerCallback(
                    optimizer,
                    **config['lr_scheduler']
                    )
    training_loop = CustomTrainingLoop(
            model,
            dataset,
            loss_fn,
            optimizer,
            **config['training'],
            seed=SEED,
            device=device,
            num_workers=16,
            val_metrics={'l1': l1, 'mse': mse},
            callbacks=[
                lr_scheduler,
                ProgressbarCallback(
                    epochs=epochs,
                    width=20),
                wandb_callback,
                checkpoint_callback,
                sanity_check_callback,
                # anomaly_detection,
                ],
            # filter_fn=partial(filter_scores, filter_threshold),
            )
    # model = training_loop.run(epochs=epochs)
    # # torch.autograd.set_detect_anomaly(True)

    checkpoint = download_wandb_checkpoint('marco-pampaloni/napoleon-zero-pytorch/bjymd6lu', 'checkpoint.pt', device=device)
    training_loop.load_state(model, checkpoint)
    # lr_scheduler.reset()
    model = training_loop.run(epochs=epochs)

    # wandb_callback.init()
    # serialize(model, training_loop, f'{ARTIFACTS_PATH}/script.pt', optimize=False)

# def serialize(model, training_loop, path, optimize=True):
#     # torch._C._jit_set_profiling_executor(False)
#     model.eval()
# 
#     serialized_model = torch.jit.script(model)
#     if optimize:
#         serialized_model = optimize_for_mobile(
#                 serialized_model,
#                 optimization_blocklist=set([MobileOptimizerType.INSERT_FOLD_PREPACK_OPS, MobileOptimizerType.HOIST_CONV_PACKED_PARAMS]),
#                 backend='CPU')
# 
#     serialized_model = torch.jit.freeze(serialized_model)
#     serialized_model = torch.jit.optimize_for_inference(serialized_model)
#     serialized_model.save(path)
#     save_wandb_file(path)
#     return serialized_model


def evaluate(sanity_check_callback, training_loop):
    print(f'Test dataset evaluation: {training_loop.evaluate()}')
    print('Sanity check positions evaluation:')
    sanity_check_callback.on_train_epoch_end(training_loop)

if __name__ == '__main__':
    main()