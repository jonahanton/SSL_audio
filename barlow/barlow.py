"""
Implementation of Barlow Twins [Zbontar et al., 2021], 
adapted from
	https://github.com/MaxLikesMath/Barlow-Twins-Pytorch
	https://github.com/facebookresearch/barlowtwins
using some code from
	https://github.com/facebookresearch/dino
	https://github.com/lucidrains/byol-pytorch
	https://github.com/yaox12/BYOL-PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn

import time
import datetime
import numpy as np

from utils import utils
from data_manager.audioset import AudioSetLoader
from models.mst import get_mst_model


class BarlowTwinsTrainer:

	def __init__(cfg):

		self.cfg = cfg
		self.time_stamp = self.cfg.checkpoint.get('time_stamp',
			datetime.datetime.now().strftime('%m%d_%H-%M'))

		if torch.cuda.is_available():
			self.device = torch.device(f'cuda:{self.cfg.local_rank}')
			torch.cuda.set_device(self.device)
			cudnn.benchmark = True
		else:
			self.device = torch.device('cpu')
		self.contruct_model()

		# checkpoint path
		self.ckpt_path = self.cfg.checkpoint.ckpt_path.format(
			self.time_stamp, self.cfg.model.encoder.type, {}
		)

		# logging
		self.logging = utils.get_std_logging()


	def construct_model(self):

		"""*****data loader*****"""
		self.data_loader = AudioSetLoader(self.cfg).get_loader() 

		"""*****build model*****"""
		if self.cfg.model.encoder.type == 'transformer':
			backbone = get_mst_model(
				size=self.cfg.model.encoder.size,
				patch_size=(self.cfg.model.encoder.ps[0], self.cfg.model.encoder.ps[1]),
			)
			if self.cfg.model.encoder.size == 'tiny':
            	embed_dim = 192 
        	elif self.cfg.model.encoder.size == 'small':
            	embed_dim = 384
        	elif self.cfg.model.encoder.size == 'base':
            	embed_dim = 768

    	if self.cfg.model.projection.sizes is None:
        	self.cfg.model.projection.sizes = [embed_dim, 4*embed_dim, 4*embed_dim, 4*embed_dim]
    
    	self.model = BarlowTwins(
        	backbone=backbone,
        	projection_sizes=self.cfg.model.projection.sizes,
        	lambd=cfg.model.lambd,
        	mask_ratio=cfg.model.encoder.mask_ratio,
		)
		# move networks to gpu
		self.model = self.model.cuda()
		# synchronize batch norms
		self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
		# ddp
		self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.cfg.gpu])
		
		"""*****prepare optimizer*****"""
		param_groups = utils.get_params_groups(self.model)
		if self.cfg.optimizer.type == 'adamw':
			self.optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
		# for mixed precision training
		fp16_scaler = None
		if cfg.meta.use_fp16:
			fp16_scaler = torch.uda.amp.GradScaler()
		
		"""*****init schedulers*****"""
		self.lr_schedule = utils.cosine_scheduler(
			base_value=self.cfg.optimizer.base_lr * (self.cfg.optimizer.batch_size / 256.),  # linear scaling rule
			final_value=self.cfg.optimizer.final_lr,
			epochs=self.cfg.optimizer.epochs, 
			niter_per_ep=len(self.data_loader),
			warmup_epochs=self.cfg.optimizer.warmup_epochs,
		)
		self.wd_schedule = utils.cosine_scheduler(
			base_value=self.cfg.optimizer.weight_decay,
			final_value=self.cfg.optimizer.final_weight_decay,
			epochs=self.cfg.optimizer.epochs,
			niter_per_ep=len(self.data_loader),
		)








def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
	
	def __init__(self, backbone, projection_sizes, lambd, mask_ratio):
		
		super().__init__()
		self.backbone = backbone
		self.lambd = lambd
		self.mask_ratio = mask_ratio
		
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
		z1 = self.projector(self.backbone(y1, mask_ratio=0.))
		z2 = self.projector(self.backbone(y2, mask_ratio=self.mask_ratio))
		
		# empirical cross-correlation matrix
		c = self.bn(z1).T @ self.bn(z2)
		
		# sum the cross-correlation matrix between all gpus
		c.div_(z1.shape[0])
		# torch.distributed.all_reduce(c)
		
		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		off_diag = off_diagonal(c).pow_(2).sum()
		loss = on_diag + self.lambd * off_diag
		return loss



