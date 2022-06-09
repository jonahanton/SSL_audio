"""
Barlow Twins for Audio (w/ Transformer encoder): Linear Evaluation.
References:
    https://github.com/nttcslab/byol-a/blob/master/evaluate.py
"""

import argparse
from pprint import pprint
import os

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import wandb

from utils import utils 
from models.mst import get_mst_model
from evaluate.linear import LinearTrainer


def get_args_parser():
    
    parser = argparse.ArgumentParser(description='Linear Evaluation', add_help=False)
    parser.add_argument('--config-path', type=str, default='./configs/config_linear_eval.yaml',
                        help='path to .yaml config file')
    return parser


def train_and_test(cfg, wandb_run):

    trainer = LinearTrainer(cfg, wandb_run)
    print(f'Starting training for {cfg.optimizer.epochs} epochs')
    for epoch in range(cfg.optimizer.epochs):
        trainer.train_one_epoch(epoch)


def eval_linear(args=None):

    if args is None:
        parser = argparse.ArgumentParser('LinearEval', parents=[get_args_parser()])
        args = parser.parse_args()

    # load training params from .ymal config file
    cfg = utils.load_yaml_config(args.config_path)
    # update config with any remaining arguments from args
    utils.update_cfg_from_args(cfg, args)

    # time stamp
    cfg.time_stamp = datetime.datetime.now().strftime('%d%m_%H-%M')

    # shared file-system initialization for torch distributed (https://pytorch.org/docs/stable/distributed.html)
    if cfg.dist_init == 'file':
        cfg.dist_url = 'file:///vol/bitbucket/jla21/proj/slurm/sharedfile'

    """set-up DDP"""
    utils.init_distributed_mode(cfg)
    # fix random seeds
    utils.fix_random_seeds(cfg.meta.seed)
    cudnn.benchmark = True

    # logging 
    print(f'Rank: {cfg.rank}')
    if cfg.rank == 0:
        wandb_run = wandb.init(
            project='BT-Audio-linear_eval',
            config=cfg,
        )
    else:
        wandb_run = None


    # run linear eval
    train_and_test(cfg, wandb_run)


if __name__ == "__main__":
    eval_linear()