"""
Barlow Twins for Audio (w/ Transformer encoder): Training.
References:
    https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    https://github.com/nttcslab/byol-a/blob/master/train.py
    https://github.com/facebookresearch/dino
"""

import argparse
from pprint import pprint
import os
import datetime
import yaml
import sys

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

import wandb

from methods import barlow, dino
from utils import utils

def get_args_parser():
    
    parser = argparse.ArgumentParser(description='Barlow Twins Training', add_help=False)
    parser.add_argument('--method', type=str, default='barlow', choices=['barlow', 'dino'],
                        help='type of ssl method to use during training')
    parser.add_argument('--config-path', type=str, default='configs/pretrain/barlow.yaml',
                        help='path to .yaml config file')
    # training hyperparameters
    parser.add_argument('-B', '--batch-size-per-gpu', type=int, default=None)
    parser.add_argument('-M', '--mask-ratio', type=float, default=None)
    parser.add_argument('-E', '--epochs', type=int, default=None)
    return parser


def train(cfg, wandb_run, logger):

    if cfg.method == 'barlow':
        trainer = barlow.BarlowTwinsTrainer(cfg, wandb_run, logger)
    elif cfg.method == 'dino':
        trainer = dino.DINOTrainer(cfg, wandb_run, logger)
    else:
        print(f'ssl method {cfg.method} is not supported. Exiting')
        sys.exit(1)
    print(f'Starting training for {cfg.optimizer.epochs} epochs')
    for epoch in range(cfg.optimizer.epochs):
        trainer.train_one_epoch(epoch)


def pretrain_btaudio(args=None):

    if args is None:
        parser = argparse.ArgumentParser('BT-A', parents=[get_args_parser()])
        args = parser.parse_args()

    # load training params from .yaml config file
    cfg = utils.load_yaml_config(args.config_path)
    # update config with any remaining arguments from args
    cfg.method = args.method
    cfg.config_path = args.config_path
    if args.batch_size_per_gpu is not None:
        cfg.optimizer.batch_size_per_gpu = args.batch_size_per_gpu
    if args.mask_ratio is not None:
        cfg.model.encoder.mask_ratio = args.mask_ratio
    if args.epochs is not None:
        cfg.optimizer.epochs = args.epochs

    # time stamp
    cfg.time_stamp = datetime.datetime.now().strftime('%d_%m')

    # update path for logging
    name = (f'{cfg.time_stamp}---method={cfg.method}-model={cfg.model.encoder.type}_{cfg.model.encoder.size}-ps={cfg.model.encoder.ps[0]}x{cfg.model.encoder.ps[1]}'
            f'-maskratio={cfg.model.encoder.mask_ratio}')
    cfg.logging.log_dir = cfg.logging.log_dir.format(name)
    cfg.checkpoint.ckpt_path = os.path.join(cfg.logging.log_dir, 'models')
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint.ckpt_path, exist_ok=True)
    # save config 
    dump = os.path.join(cfg.logging.log_dir, 'pretrain_params.yaml')
    if utils.is_main_process():
        with open(dump, 'w') as f:
            yaml.dump(cfg, f)


    """set-up DDP"""
    utils.init_distributed_mode(cfg)
    # fix random seeds
    utils.fix_random_seeds(cfg.meta.seed)
    cudnn.benchmark = True

    # logging 
    logger = utils.get_std_logging(filename=os.path.join(cfg.logging.log_dir, 'out.log')) 
    # wandb 
    if utils.is_main_process():
        wandb_run = wandb.init(
            project='BT-Audio-pretrain',
            config=cfg,
            settings=wandb.Settings(start_method="fork"),
        )
    else:
        wandb_run = None
    
    # run training
    train(cfg, wandb_run, logger)



if __name__ == "__main__":
    pretrain_btaudio()
