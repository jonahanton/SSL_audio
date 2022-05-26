"""
Barlow Twins for Audio (w/ Transformer encoder): Training.
References:
    https://github.com/nttcslab/byol-a/blob/master/train.py
"""

import os
import sys
import numpy as np
import random
from pathlib import Path
import time
import multiprocessing

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
    data_loader_train = torch.utils.data.DataLoader(
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
    backbone.to(device)

    if cfg.lr is None:  # only base_lr is specified
        cfg.lr = cfg.blr * cfg.bs / 256
    

