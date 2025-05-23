#!/usr/bin/python3

import yaml
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from datasets import BitboardDataset, AugmentedDataset
from models import BitboardTransformer 
from potatorch.training import TrainingLoop
from potatorch.callbacks import ProgressbarCallback, LRSchedulerCallback
from potatorch.callbacks import WandbCallback, CheckpointCallback, SanityCheckCallback
from datasets.BitboardDataset import string_to_matrix
from losses import MSLELoss, WDLLoss, WMSELoss, MARELoss, SMARELoss, smare_loss, mare_loss
from functools import partial
from typing import List, Callable
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

def read_positions(file: str):
    """ Read positions from `file` and return the following information:
        - descriptors (either fen of description string) list
        - positions (bitboards) tensor
        - auxiliary inputs (side, ep square, castling) tensor
        - scores tensor
    """
    dtypes: dict = {
            0: 'string',                # fen position string
            1: 'string', 2: 'string',   # bitboards
            3: 'string', 4: 'string',   # bitboards
            5: 'string', 6: 'string',   # bitboards
            7: 'string', 8: 'string',   # bitboards
            9: 'string', 10: 'string',  # bitboards
            11: 'string', 12: 'string', # bitboards
            13: 'uint8',                # side to move: 0 = white, 1 = black
            14: 'uint8',                # enpassant square: 0-63, 65 is none 
            15: 'uint8',                # castling status: integer [0000 b4 b3 b2 b1] b1 = K, b2 = Q, b3 = k, b4 = q
            16: 'int32'                 # score: integer value (mate = 2^15 - 1)
            }
    df = pd.read_csv(file, header=None, dtype=dtypes)

    fens = df.iloc[:, 0].values
    bitboards = df.iloc[:, 1:13].values
    aux = df.iloc[:, 13:-1].values
    score = df.iloc[:, -1:].values

    x = [(np.array([[string_to_matrix(b) for b in bs]])) for bs in bitboards]
    aux = [(np.array([v])) for v in aux]
    score = [(np.array([v / 100.0])) for v in score]

    return fens, x, aux, score

def rescale_bitboards(bs):
    """ Scale each bitboard by [0.1, 0.2, ..., 0.6] relatively for both colors """
    return np.array([bs[i] * (i%6 + 1) / 10 for i in np.arange(12)])

def filter_scores(filter_threshold: int, data: tuple) -> bool:
    """ Filter scores based on a given cutoff threshold """
    return abs(data[2]) < filter_threshold

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# TODO: Weight MSE by number of pieces on the board: x.sum() should reduce the tensor to the total number of pieces
# This should prioritize early game (tactical) positions.
# Another option (which might be used alongside this one) is to map the scores in the [-1, 1] interval using tanh, while
# still using MSELoss.
def make_loss_fn(name: str, config: dict) -> nn.Module:
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'wmse':
        return WMSELoss()
    elif name == 'wdl':
        if 'logits_target_scale' not in config.get('general', {}).keys():
            raise Exception("When specifying 'wdl' as loss you need to provide `logits_target_scale`")

        target_scale = config.get('general', {}).get('target_scale', 1.0)

        # scaling the inputs of the WDLLoss is used to drive the output of the sigmoid (target) to be close to one When
        # the white has a winning position with high probability and close to zero when black is winning with high
        # probability. If dividing by `target_scale` the targets rescales the score so that 1.0 = 1 pawn advantage, then
        # using a `logits_target_scale` of 0.4 means that the sigmoid will output a probability of white winning of
        # about 0.9. Making `logits_target_scale` smaller means requiring larger scores to have a high probability of
        # winning.
        s = config['general']['logits_target_scale'] / target_scale
        return WDLLoss(s, s)
    elif name == 'huber':
        return nn.HuberLoss(delta=1.0)
    elif name == 'mare':
        return MARELoss()
    elif name == 'smare':
        return SMARELoss()
    elif name == 'msle':
        if 'mate_value' not in config.get('general', {}).keys():
            raise Exception('When specifying `msle` as loss you need to provide `mate_value`')

        target_scale = config.get('general', {}).get('target_scale', 1.0)
        mate_value = config['general']['mate_value']
        return MSLELoss(delta=mate_value*target_scale)

    raise Exception(f"{name}: Loss not implemented yet.")

class CustomTrainingLoop(TrainingLoop):
    def __init__(self, *args, input_features = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_features = input_features

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
        if self.input_features:
            loss = self.loss_fn(X, pred.view(-1), ys)
        else:
            loss = self.loss_fn(pred.view(-1), ys)
        return loss

def rotate_board(bitboards, aux, score):
    """ Rotate the board 180 degrees, switching the colors of pieces, side to move, castling status and en-passant
    square """
    # Bitboards are shaped 12x8x8
    wpieces = bitboards[:6]
    bpieces = bitboards[6:]

    # Side to move is 0 if it's white's turn, 1 otherwise
    stm = (int(aux[0]) + 1) % 2                         # switch side to move

    # 0-63 for valid squares, 65 for invalid ones (64 is unused)
    ep0 = int(aux[1])
    ep = 63 - ep0 if 0 <= ep0 < 64 else ep0    # rotate en passant square

    if ep0 < 0:
        print(bitboards)
        print(aux)
        print(score)
        raise Exception('Invalid enpassant square')

    # Castling is encoded in the 4 least significant bits of a byte:
    # b4 b3 b2 b1 -> We swap b4 with b2 and b3 with b1 (0xC = 1100, 0x3 = 0011).
    castle = (int(aux[2]) & 0xC) >> 2 | (int(aux[2]) & 0x3) << 2 # swap castling white <-> castling black

    return torch.cat([bpieces, wpieces], dim=0).flip(-2, -1), torch.tensor([stm, ep, castle]).float(), -score

def map_augmentations(augmentations: List[str]) -> List[Callable]:
    transformations = []
    for fname in augmentations:
        if fname == 'rotation':
            transformations.append(rotate_board)
        else:
            raise Exception('Undefined augmentation')
    return transformations

def main():
    config = load_config(f'{sys.path[0]}/../config/vit_training_baseline_augmented.yaml')

    SEED = config['general']['seed']
    DRIVE_PATH = f'{sys.path[0]}/../datasets/datasets'
    ARTIFACTS_PATH = f'{sys.path[0]}/../artifacts'

    set_random_state(SEED)

    # Get cpu or gpu device for training.
    device: str = config['general']['device']
    print(f"Using {device} device")

    torch.backends.cuda.matmul.allow_tf32 = True # TODO: why?

    (check_fen, check_x, check_aux, check_score) = read_positions(f'{sys.path[0]}/sanity_check.csv')

    target_scale = config['general']['target_scale'] # further rescale the target score so that it's easier to train on
    filter_threshold = config.get('general', {}).get('filter_threshold', None)

    if filter_threshold:
        filter_threshold *= target_scale

    # NOTE: Stockfish evaluates some deep mating positions to ±200 probably because of endgame tables so we clamp scores
    # at mate_value
    mate_value = config.get('general', {}).get('mate_value', 30)
    dataset = BitboardDataset(
                dir=DRIVE_PATH,
                seed=SEED,
                glob=False,
                preload=False,
                from_dump=False,
                low_memory=True,
                fraction=1.0,
                target_transform=(lambda y: np.clip(y, -mate_value, mate_value) * target_scale), # scores in the dataset are expressed in centipawns / 100
                debug=True,
                **config['dataset'],
                # transform=(lambda x, aux: (rescale_bitboards(x), aux)),
                )
    config['dataset']['size'] = len(dataset)

    # TODO: multiplying bitboards planes seems to be helpful (e.g. pawn * 0.1, bishop * 0.2, etc.)
    # TODO: try training first with MSE and do a round of post-training optimization with a different loss
    # (e.g. binary cross-entropy loss)
    # TODO: try multiplicative conditioning with auxiliary input instead of concatenating it to the input as an
    # additional channel
    # TODO: Unwrapping the bits from the castling status of the auxiliary input might be beneficial (though the number
    # of configurations is just 2^4 = 16, so it should easy enough for the network to discriminate each case)

    model = BitboardTransformer(**config['model'])
    model = model.to(device, memory_format=torch.channels_last) #type: ignore
    print_summary(model)

    # TODO: try gradient clipping
    optimizer = torch.optim.AdamW(
            model.parameters(),
            **{**config['optimizer'], 'betas': tuple(config.get('optimizer', {}).get('betas', [0.9, 0.999]))}
            )

    sanity_check_callback = SanityCheckCallback(
            data=list(zip(check_x, check_aux, check_score)),
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

    l1 = lambda pred, inputs: F.l1_loss(pred.view(-1), inputs[-1])
    mse = lambda pred, inputs: F.mse_loss(pred.view(-1), inputs[-1])
    mare = lambda pred, inputs: mare_loss(pred.view(-1), inputs[-1])
    smare = lambda pred, inputs: smare_loss(pred.view(-1), inputs[-1])

    epochs = config['general']['epochs']
    loss_name = config['general']['loss_function']
    loss_fn = make_loss_fn(loss_name, config)
    lr_scheduler = LRSchedulerCallback(
                    optimizer,
                    **config['lr_scheduler']
                    )

    augmentations = config['general'].get('augmentations', [])
    augmenter = None
    if augmentations:
        augmenter = lambda dataset: AugmentedDataset(dataset, transforms=map_augmentations(augmentations))

    training_loop = CustomTrainingLoop(
        model,
        dataset,
        loss_fn,
        optimizer,
        **config['training'],
        input_features=(loss_name == 'wmse'),
        seed=SEED,
        device=device,
        num_workers=12,
        augmenter=augmenter,
        val_metrics={'l1': l1, 'mse': mse, 'mare': mare, 'smare': smare},
        callbacks=[
            lr_scheduler,
            ProgressbarCallback(
                epochs=epochs,
                width=20),
            wandb_callback,
            checkpoint_callback,
            sanity_check_callback,
        ],
        filter_fn=partial(filter_scores, filter_threshold) if filter_threshold else None,
    )
    # checkpoint = download_wandb_checkpoint('marco-pampaloni/napoleon-zero-pytorch/86woan88', 'checkpoint.pt',
    #                                        device=device, replace=True)
    # training_loop.load_state(model, checkpoint)
    # lr_scheduler.reset()
    model = training_loop.run(epochs=epochs)

def evaluate(training_loop):
    print(f'Test dataset evaluation: {training_loop.evaluate(metrics=training_loop.val_metrics)}')
    # print('Sanity check positions evaluation:')
    # sanity_check_callback.on_train_epoch_end(training_loop)

if __name__ == '__main__':
    main()