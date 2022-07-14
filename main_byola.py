import argparse
import os
import datetime
import yaml
import wandb 
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from utils import utils
from byola.train import BYOLATrainer

def get_args_parser():
    
    parser = argparse.ArgumentParser(description='BYOL-A', add_help=False)
    parser.add_argument('--config-path', type=str, default='byola/config.yaml',
                        help='path to .yaml config file')
    return parser

def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser('BYOL-A', parents=[get_args_parser()])
        args = parser.parse_args()

    # load training params from .yaml config file
    cfg = utils.load_yaml_config(args.config_path)
    # update config with any remaining arguments from args
    cfg.config_path = args.config_path

    # time stamp
    cfg.time_stamp = datetime.datetime.now().strftime('%d_%m')

    # update path for logging
    name = (f'{cfg.time_stamp}---method=byola-model={cfg.model.encoder.type}')
    cfg.logging.log_dir = cfg.logging.log_dir.format(name)
    cfg.checkpoint.ckpt_path = os.path.join(cfg.logging.log_dir, 'models')
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint.ckpt_path, exist_ok=True)

    """set-up DDP"""
    utils.init_distributed_mode(cfg)
    # fix random seeds
    utils.fix_random_seeds(cfg.meta.seed)
    cudnn.benchmark = True

    # save config 
    dump = os.path.join(cfg.logging.log_dir, 'pretrain_params.yaml')
    if utils.is_main_process():
        with open(dump, 'w') as f:
            yaml.dump(cfg, f)

    # logging 
    logger = utils.get_std_logging(filename=os.path.join(cfg.logging.log_dir, 'out.log')) 
    # wandb 
    if utils.is_main_process():
        wandb_run = wandb.init(
            project='byola',
            config=cfg,
            settings=wandb.Settings(start_method="fork"),
        )
    else:
        wandb_run = None
    
    # run byol-a
    byola(cfg, wandb_run, logger)


def byola(cfg, wandb_run, logger):

    trainer = BYOLATrainer(cfg, wandb_run, logger)
    print(f'Starting training for {cfg.pretrain.optimizer.epochs} epochs')
    for epoch in range(cfg.pretrain.optimizer.epochs):
        trainer.train_one_epoch(epoch)


if __name__ == "__main__":
    main()