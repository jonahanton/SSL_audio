"""
Linear evaluation of pre-trained model on AudioSet-20K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn

import time
import datetime
import numpy as np
import sys
import math
import os
from pathlib import Path
from pprint import pprint 
import json

from utils import utils
from data_manager.audioset import AudioSetLoader
from models.mst import get_mst_model

class LinearTrainer:

	def __init__(self, cfg, wandb_run):
		
		self.cfg = cfg
		self.wandb_run = wandb_run

		self.construct_model()

    
    def construct_model(self):
        
        """*****data loaders*****"""
		print(f'Loading AudioSet-20K')
		self.data_loader_train = AudioSetLoader(cfg=self.cfg, pretrain=False).get_loader() 
		print(f'Loaded AudioSet-20K, with {len(self.data_loader_train) * self.cfg.optimizer.batch_size_per_gpu * self.cfg.world_size} data points')
        
        print(f'Loading AudioSet evaluation set')
        self.data_loader_test = AudioSetLoader(cfg=self.cfg, pretrain=False).get_loader(test=True) 
        print(f'Loaded AudioSet evaluation set, with {len(self.data_loader_test) * self.cfg.optimizer.batch_size_per_gpu * self.cfg.world_size} data points')

        
        """*****build model*****"""
        if cfg.model.encoder.type == 'transformer':
            backbone = get_mst_model(
                size=cfg.model.encoder.size,
                patch_size=(cfg.model.encoder.ps[0], cfg.model.encoder.ps[1])
            )
            embed_dim = backbone.embed_dim
        
        # Load pre-trained weights 
        if cfg.weight_file is not None:
            utils.load_pretrained_weights(model=backbone, weight_file=cfg.weight_file)
        



