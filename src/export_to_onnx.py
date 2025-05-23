#!/usr/bin/python3

import torch
from models import BitboardTransformer 
from potatorch.utils import load_model_from_wandb_checkpoint
from utils import export_onnx
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str, help='Wandb run id from which to download the model')
    return parser.parse_args()

def make_model(config: dict, device: str = 'cpu'):
    return BitboardTransformer(**config['model']).to(device, memory_format=torch.channels_last)

def main(args):
    run_id = args.run_id
    model = load_model_from_wandb_checkpoint(f'napoleon-zero-pytorch/{run_id}', make_model, device='cpu')
    export_onnx(model)

if __name__ == '__main__':
    args = parse_args()
    if not args.run_id:
        raise ValueError("No run id provided")
    main(args)
