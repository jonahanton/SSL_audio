"""
Barlow Twins for Audio (w/ Transformer encoder): Training.
References:
    https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    https://github.com/nttcslab/byol-a/blob/master/train.py
"""

import os
import sys
import numpy as np
import random
from pathlib import Path
import time
import multiprocessing
import math

import torch
from torch import nn, optim

from barlow.barlow import BarlowTwins
from data_manager.audioset import AudioSet
from data_manager.transforms import make_transforms
from models.mst import get_mst_model
from utils.misc import load_yaml_config, LARS


def main(config_path):
    
    cfg = load_yaml_config(config_path)
    device = torch.device(cfg.device)

    # data preparation
    wav_transform, lms_transform = make_transforms(cfg)
    audioset_dataset = AudioSet(cfg, wav_transform=wav_transform, lms_transform=lms_transform)
    loader = torch.utils.data.DataLoader(
                            audioset_dataset, 
                            batch_size=cfg.bs,
                            num_workers=multiprocessing.cpu_count(),
                            pin_memory=True, 
                            shuffle=True,
                            drop_last=True,
                        )
    
    # encoder backbone
    if cfg.encoder_type == 'transformer':
        backbone = get_mst_model(
            size=cfg.encoder_size,
            patch_size=(cfg.encoder_ps[0], cfg.encoder_ps[1])
        )
        # embed dim
        if cfg.encoder_size == 'tiny':
            embed_dim = 192 
        elif cfg.encoder_size == 'small':
            embed_dim = 384
        elif cfg.encoder_size == 'base':
            embed_dim = 768
    
    # barlow twins
    if cfg.projection_sizes is None:
        cfg.projection_sizes = [embed_dim, 4*embed_dim, 4*embed_dim, 4*embed_dim]
    
    bt_model = BarlowTwins(
        backbone=backbone,
        projection_sizes=cfg.projection_sizes,
        lambd=cfg.lambd,
        mask_ratio=cfg.encoder_mask_ratio,
    )
    bt_model.to(device)

    param_weights = []
    param_biases = []
    for param in bt_model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(
        parameters,
        lr=0, 
        weight_decay=cfg.weight_decay,
        weight_decay_filter=True,
        lars_adaptation_filter=True,
    )


    # training loop
    for epoch in range(cfg.epochs):
        for step, (y, y_tf) in enumerate(loader, start=epoch * len(loader)):
            y = y.to(device)
            y_tf = y_tf.to(device)
            
            adjust_learning_rate(cfg, optimizer, loader, step)
            
            optimizer.zero_grad()
            loss = bt_model.forward(y, y_tf)
            loss.backward()
            optimizer.step()




def adjust_learning_rate(cfg, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = cfg.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * cfg.lr_w
    optimizer.param_groups[1]['lr'] = lr * cfg.lr_b



if __name__ == "__main__":

    config_path = 'config.yaml'
    main(config_path)
