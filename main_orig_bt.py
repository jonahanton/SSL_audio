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


def get_args_parser():
    
    parser = argparse.ArgumentParser(description='Barlow Twins Training', add_help=False)
    parser.add_argument('--config-path', type=str, default='./configs/orig_bt.yaml',
                        help='path to .yaml config file')
    return parser

def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser('BT-A-r50', parents=[get_args_parser()])
        args = parser.parse_args()

    # load training params from .yaml config file
    cfg = utils.load_yaml_config(args.config_path)
    # update config with any remaining arguments from args
    cfg.config_path = args.config_path

    # time stamp
    cfg.time_stamp = datetime.datetime.now().strftime('%d_%m')

    # update path for logging
    name = (f'{cfg.time_stamp}---model={cfg.model.encoder.type}')
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
            project='BT-Audio-pretrain-resnet',
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

    """*****build model*****"""
    embed_dim = 2048
    if cfg.model.projection.sizes is None:
        p_x = cfg.model.projection.projector_x 
        cfg.model.projection.sizes = [embed_dim, p_x*embed_dim, p_x*embed_dim, p_x*embed_dim]
    
    model = BarlowTwins(
        cfg=cfg,
        projection_sizes=cfg.model.projection.sizes,
        lambd=cfg.model.lambd,
    )
    # move model to gpu
    model = model.cuda(cfg.gpu)
    
    if cfg.meta.distributed:
        # synchronize batch norms
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    """*****prepare optimizer*****"""
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(parameters, lr=0, weight_decay=1e-6,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    if cfg.meta.distributed:
        # wrap model with ddp
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.gpu],
            output_device=cfg.gpu,
        )
        model_without_ddp = model.module

    """*****init schedulers*****"""
    lr_schedule_weights = utils.cosine_scheduler(
        base_value=cfg.optimizer.lr_weights * (cfg.optimizer.batch_size_per_gpu * cfg.world_size / 256.),  # linear scaling rule
        final_value=cfg.optimizer.lr_weights * 0.001,
        epochs=cfg.optimizer.epochs, 
        niter_per_ep=len(data_loader),
        warmup_epochs=cfg.optimizer.warmup_epochs,
    )
    lr_schedule_biases = utils.cosine_scheduler(
        base_value=cfg.optimizer.lr_biases * (cfg.optimizer.batch_size_per_gpu * cfg.world_size / 256.),  # linear scaling rule
        final_value=cfg.optimizer.lr_biases * 0.001,
        epochs=cfg.optimizer.epochs, 
        niter_per_ep=len(data_loader),
        warmup_epochs=cfg.optimizer.warmup_epochs,
    )


    start_epoch = 0
    """*****run training*****"""
    for epoch in range(start_epoch, cfg.optimizer.epochs):

        metric_logger = utils.MetricLogger(delimiter=" ")
        header = f'Epoch: [{epoch}/{cfg.optimizer.epochs}]'

        # knn mAP metric
        track_knn = cfg.knn.track_knn and (epoch % cfg.knn.track_knn_it == 0)

        end = time.time()
        for iteration, ((y1, y2), labels) in enumerate(metric_logger.log_every(data_loader, cfg.checkpoint.print_it, header)):
            # measure data loading time
            metric_logger.update(data_time=(time.time()-end))

			# update weight decay and learning rate according to their schedule 
            iteration = len(data_loader) * epoch + iteration  # global training iteration

            optimizer.param_groups[0]["lr"] = lr_schedule_weights[iteration]
            optimizer.param_groups[1]["lr"] = lr_schedule_biases[iteration]
            
            # move to gpu
            y1 = y1.cuda(non_blocking=True)
            y2 = y2.cuda(non_blocking=True)
            
            tflag = time.time()
            # forward passes + compute barlow twins loss
            loss = model(y1, y2)
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
            lr_weights = optimizer.param_groups[0]["lr"]
            lr_biases = optimizer.param_groups[1]["lr"]
            metric_logger.update(loss=loss_val)
            metric_logger.update(lr_weights=lr_weights)
            metric_logger.update(lr_biases=lr_biases)

            if wandb_run is not None:
                wandb_run.log({
                    'train_loss': metric_logger.meters['loss'].value,
					'lr_weights': lr_weights,
                    'lr_biases': lr_biases,
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
            knn_mAP = knn_metric.predict_knn(cfg, model.module.backbone, knn_train_loader, knn_test_loader)
            
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
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'epoch': epoch,
                'config': cfg,
            }
            utils.save_on_master(save_dict, ckpt_path.format(f'epoch-{epoch}'))


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class BarlowTwins(nn.Module):
    def __init__(self, cfg, projection_sizes, lambd):
        super().__init__()
        self.cfg = cfg
        self.backbone = torchvision.models.resnet50(zero_init_residual=True, pretrained=False)
        self.backbone.fc = nn.Identity()
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)  # spectrogram inputs have only 1 channel
        self.lambd = lambd

        # projector
        sizes = projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0])
        if self.cfg.meta.distributed:
            torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



if __name__ == "__main__":
    main()