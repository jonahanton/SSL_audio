"""
Barlow Twins for Audio (w/ Transformer encoder): Training.
References:
    https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    https://github.com/nttcslab/byol-a/blob/master/train.py
"""

import argparse
from pprint import pprint

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from barlow.barlow import BarlowTwinsTrainer
from utils import utils

def get_args_parser():
    
    parser = argparse.ArgumentParser(description='Barlow Twins Training', add_help=False)
    parser.add_argument('-cp', '--config-path', type=str, default='./config.yaml',
                        help='path to .yaml config file')
    
    return parser


def train_bt(args):

    # load training params from .ymal config file
    cfg = utils.load_yaml_config(args.config_path)
    # update config with any remaining arguments from args
    utils.update_cfg_from_args(cfg, args)

    # update path for logging
    name = (f'{cfg.model.encoder.type}-ps{cfg.model.encoder.ps[0]}x{cfg.model.encoder.ps[1]}'
            f'-maskratio{cfg.model.encoder.mask_ratio}')
    cfg.logging.log_path = cfg.logging.log_path.format(name)

    # set up ddp
    utils.init_distributed_mode(cfg)
    # fix random seeds
    utils.fix_random_seeds(cfg.meta.seed)
    cudnn.benchmark = True

    trainer = BarlowTwinsTrainer(cfg)
    for epoch in range(cfg.optimizer.epochs):
        trainer.train_one_epoch(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('BT-A', parents=[get_args_parser()])
    args = parser.parse_args()
    train_bt(args)
