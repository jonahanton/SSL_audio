"""
Barlow Twins for Audio (w/ Transformer encoder): Linear Evaluation.
References:
    https://github.com/nttcslab/byol-a/blob/master/evaluate.py
"""

import argparse
from pprint import pprint
import os
import datetime
import yaml 
import json

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import wandb

from utils import utils 
from evaluate.linear import LinearTrainer


def get_args_parser():
    
    parser = argparse.ArgumentParser(description='Linear Evaluation', add_help=False)
    parser.add_argument('--config-path', type=str, default='./configs/lineval/config.yaml',
                        help='path to .yaml config file')
    parser.add_argument('-w', '--weight-file', type=str, default=None)
    return parser


def train_and_test(cfg, wandb_run, logger):

    trainer = LinearTrainer(cfg, wandb_run, logger)
    print(f'Starting training for {cfg.optimizer.epochs} epochs')
    for epoch in range(cfg.optimizer.epochs):
        trainer.train_one_epoch(epoch)


def eval_linear(args=None):

    if args is None:
        parser = argparse.ArgumentParser('LinearEval', parents=[get_args_parser()])
        args = parser.parse_args()

    # load training params from .yaml config file
    cfg = utils.load_yaml_config(args.config_path)
    # update config with any remaining arguments from args
    cfg.config_path = args.config_path
    if args.weight_file is not None:
        cfg.weight_file = args.weight_file
    model_base_name = (cfg.weight_file.split('/')[-1]).replace('.pth.tar', '')

    # time stamp
    cfg.time_stamp = datetime.datetime.now().strftime('%d%m_%H-%M')

    # set-up path for logging
    if cfg.logging.log_dir is None:
        cfg.logging.log_dir = '/'.join(cfg.weight_file.split('/')[:-2]) + '/lineval/' + model_base_name 
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    # save config 
    dump = os.path.join(cfg.logging.log_dir, 'lineval_params.yaml')
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
            project='BT-Audio-lineval',
            config=cfg,
            settings=wandb.Settings(start_method="fork"),
        )
    else:
        wandb_run = None

    # run linear eval
    train_and_test(cfg, wandb_run, logger)


if __name__ == "__main__":
    eval_linear()