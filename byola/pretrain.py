"""
Re-implementation of BYOL for Audio [Niizumi et al., 2021].
Code adapted from https://github.com/nttcslab/byol-a.
"""

import argparse
from pprint import pprint
import os
import datetime
import time
import yaml
import wandb 
import numpy as np
import sys 
import math 
import json 
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp 
import torchvision

from utils import utils, knn_metric
from data_manager.audioset import AudioSetLoader
from data_manager.audioset_lms import SpectrogramLoader

from byola.byol_pytorch import BYOL

def get_args_parser():
    
    parser = argparse.ArgumentParser(description='BYOL-A Training', add_help=False)
    parser.add_argument('--config-path', type=str, default='byola/configs/pretrain.yaml',
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
            project='byola-pretrain',
            config=cfg,
            settings=wandb.Settings(start_method="fork"),
        )
    else:
        wandb_run = None
    
    # run training
    train(cfg, wandb_run, logger)


def train(cfg, wandb_run, logger):
    
    ckpt_path = os.path.join(cfg.checkpoint.ckpt_path, '{}.pth.tar')
    
    """*****data loaders*****"""
    print(f'Loading AudioSet')
    utils.log_on_master(logger, f'Loading AudioSet')
	
    if cfg.data.dataloader.npy:
        # Load in pre-converted raw waveforms (.wav) -> lms (.npy) files 
        data_loader = SpectrogramLoader(
            cfg,
            pretrain=True,
            balanced_only=cfg.data.audioset.balanced_only,
            ).get_loader()
    else:
        # Load in raw waveforms (.wav)
        data_loader = AudioSetLoader(
            cfg,
            pretrain=True,
            balanced_only=cfg.data.audioset.balanced_only,
            ).get_loader() 
    print(f'Loaded AudioSet, with {len(data_loader.dataset)} data points')
    utils.log_on_master(logger, f'Loaded AudioSet, with {len(data_loader.dataset)} data points')

    """*****build BYOL learner*****"""
    learner = BYOL(
        net=AudioNTT2020(
            n_mels=cfg.data.preprocess.n_mels,
            d=cfg.model.encoder.feature_d,
        ),
        image_size=cfg.data.preprocess.shape,
        projection_size=cfg.model.projection.proj_size,
        projection_hidden_size=cfg.model.projection.proj_dim,
        moving_average_decay=cfg.model.ema_decay,
    )
    # move to gpu 
    learner = learner.cuda(cfg.gpu)
    if cfg.meta.distributed:
        # synchronize batch norms
        learner = nn.SyncBatchNorm.convert_sync_batchnorm(learner)
        # wrap with ddp
        learner = nn.parallel.DistributedDataParallel(
            learner,
            device_ids=[cfg.gpu],
            output_device=cfg.gpu,
        )
        learner_without_ddp = learner.module
    else:
        learner_without_ddp = learner

    """*****prepare optimizer*****"""
    optimizer = torch.optim.Adam(learner.parameters(), lr=cfg.optimizer.lr)

    """*****run training*****"""
    start_epoch = 0
    for epoch in range(start_epoch, cfg.optimizer.epochs):

        metric_logger = utils.MetricLogger(delimiter=" ")
        header = f'Epoch: [{epoch}/{cfg.optimizer.epochs}]'

        # knn mAP metric
        track_knn = cfg.knn.track_knn and (epoch % cfg.knn.track_knn_it == 0)

        end = time.time()
        for iteration, ((y1, y2), labels) in enumerate(metric_logger.log_every(data_loader, cfg.checkpoint.print_it, header)):
            # measure data loading time
            metric_logger.update(data_time=(time.time()-end))
            
            # move to gpu
            y1 = y1.cuda(non_blocking=True)
            y2 = y2.cuda(non_blocking=True)
            
            tflag = time.time()
            # forward passes + compute loss
            loss = learner(y1, y2)
            metric_logger.update(forward_time=time.time()-tflag)
            
            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training")
                sys.exit(1)

            tflag = time.time()
            # gradient update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            metric_logger.update(backward_time=(time.time()-tflag))
            
            # logging 
            if cfg.meta.distributed:
                torch.cuda.synchronize()
            loss_val = loss.item()
            metric_logger.update(loss=loss_val)

            if wandb_run is not None:
                wandb_run.log({
                    'train_loss': metric_logger.meters['loss'].value,
                    'data_time' : metric_logger.meters['data_time'].value,
                    'forward_time' : metric_logger.meters['forward_time'].value,
                    'backward_time' : metric_logger.meters['backward_time'].value,
                })
            
            end = time.time()


        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print(f'Averaged stats: {metric_logger}')
        utils.log_on_master(logger, f'Averaged stats: {metric_logger}')

        # return training stats
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # knn mAP metric
        if track_knn:
            
            print('Calculating knn mAP')
            utils.log_on_master(logger, 'Calculating knn mAP')
            # obtain data loaders
            if cfg.data.dataloader.npy:
                knn_train_loader = SpectrogramLoader(cfg, pretrain=False, balanced_only=True,test=False).get_loader(drop_last=False)
                knn_test_loader = SpectrogramLoader(cfg, pretrain=False, test=True).get_loader(drop_last=False)
            else:
                knn_train_loader = AudioSetLoader(cfg, pretrain=False, balanced_only=True,test=False).get_loader(drop_last=False)
                knn_test_loader = AudioSetLoader(cfg, pretrain=False, test=True).get_loader(drop_last=False)
            # extract features + calculate knn mAP
            knn_mAP = knn_metric.predict_knn(cfg, learner_without_ddp.online_encoder, knn_train_loader, knn_test_loader)
            
            print(f'knn mAP: {knn_mAP}')
            utils.log_on_master(logger, f'knn mAP: {knn_mAP}')
            train_stats.update({'knn_mAP': knn_mAP})
            
            if wandb_run is not None:
                wandb_run.log({
                    'test_knn_mAP': knn_mAP,
                })
		
        # save epoch logs
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
        }
        if utils.is_main_process():
            with (Path(f'{cfg.logging.log_dir}/log.txt')).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
        # save checkpoint
        if (epoch % cfg.checkpoint.save_epoch_it == 0) or (epoch == cfg.optimizer.epochs - 1):
            save_dict = {
                'model': learner.state_dict(),
                'opt': optimizer.state_dict(),
                'epoch': epoch,
                'config': cfg,
            }
            utils.save_on_master(save_dict, ckpt_path.format(f'epoch-{epoch}'))


class NetworkCommonMixIn():
    """Common mixin for network definition."""

    def load_weight(self, weight_file, device):
        """Utility to load a weight file to a device."""

        state_dict = torch.load(weight_file, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Remove unneeded prefixes from the keys of parameters.
        weights = {}
        for k in state_dict:
            m = re.search(r'(^fc\.|\.fc\.|^features\.|\.features\.)', k)
            if m is None: continue
            new_k = k[m.start():]
            new_k = new_k[1:] if new_k[0] == '.' else new_k
            weights[new_k] = state_dict[k]
        # Load weights and set model to eval().
        self.load_state_dict(weights)
        self.eval()
        print(f'Using audio embbeding network pretrained weight: {Path(weight_file).name}')
        return self

    def set_trainable(self, trainable=False):
        for p in self.parameters():
            p.requires_grad = trainable



class AudioNTT2020Task6(nn.Module, NetworkCommonMixIn):
    """DCASE2020 Task6 NTT Solution Audio Embedding Network."""

    def __init__(self, n_mels, d):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.d = d

    def forward(self, x):
        x = self.features(x)       # (batch, ch, mel, time)       
        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
        x = self.fc(x)
        return x


class AudioNTT2020(AudioNTT2020Task6):
    """BYOL-A General Purpose Representation Network.
    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.
    """

    def __init__(self, n_mels=64, d=512):
        super().__init__(n_mels=n_mels, d=d)

    def forward(self, x):
        x = super().forward(x)
        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2
        assert x.shape[1] == self.d and x.ndim == 2
        return x


if __name__ == "__main__":
    main()


