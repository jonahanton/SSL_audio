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
import torch.distributed as dist

import time
import datetime
import numpy as np
import sys
import math
import os
from pathlib import Path
from pprint import pprint 
import json

from utils import utils, knn_metric
from data_manager.audioset import AudioSetLoader
from data_manager.audioset_lms import SpectrogramLoader
from models.mst import get_mst_model


class BarlowTwinsTrainer:

	def __init__(self, cfg, wandb_run, logger):
		
		self.cfg = cfg
		self.wandb_run = wandb_run
		self.logger = logger
		# checkpoint path
		self.ckpt_path = os.path.join(self.cfg.checkpoint.ckpt_path, '{}.pth.tar')
		self.construct_model()

		print(f'Config parameters: \n{self.cfg}')
		utils.log_on_master(self.logger, f'Config parameters: \n{self.cfg}')

	def construct_model(self):

		"""*****data loaders*****"""
		print(f'Loading AudioSet')
		utils.log_on_master(self.logger, f'Loading AudioSet')
		
		if self.cfg.data.dataloader.npy:
			# Load in pre-converted raw waveforms (.wav) -> lms (.npy) files 
			self.data_loader = SpectrogramLoader(
				self.cfg,
				pretrain=True,
				balanced_only=self.cfg.data.audioset.balanced_only,
			).get_loader()
		else:
			# Load in raw waveforms (.wav)
			self.data_loader = AudioSetLoader(
				self.cfg,
				pretrain=True,
				balanced_only=self.cfg.data.audioset.balanced_only,
			).get_loader() 
		print(f'Loaded AudioSet, with {len(self.data_loader.dataset)} data points')
		utils.log_on_master(self.logger, f'Loaded AudioSet, with {len(self.data_loader.dataset)} data points')

		"""*****build model*****"""
		if self.cfg.model.encoder.type == 'transformer':
			backbone = get_mst_model(
				size=self.cfg.model.encoder.size,
				patch_size=(self.cfg.model.encoder.ps[0], self.cfg.model.encoder.ps[1])
			)
			embed_dim = backbone.embed_dim
			
		if self.cfg.model.projection.sizes is None:
			p_x = self.cfg.model.projection.projector_x 
			self.cfg.model.projection.sizes = [embed_dim, p_x*embed_dim, p_x*embed_dim, p_x*embed_dim]
	
		self.model = BarlowTwins(
			cfg=self.cfg,
			backbone=backbone,
			projection_sizes=self.cfg.model.projection.sizes,
			lambd=self.cfg.model.lambd,
			mask_ratio=self.cfg.model.encoder.mask_ratio,
		)
		# move model to gpu
		self.model = self.model.cuda(self.cfg.gpu)
		# if self.cfg.meta.distributed:
		# synchronize batch norms
		self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
		# wrap model with ddp
		self.model = nn.parallel.DistributedDataParallel(
			self.model,
			device_ids=[self.cfg.gpu],
			output_device=self.cfg.gpu,
		)
		self.model_without_ddp = self.model.module
		
		"""*****prepare optimizer*****"""
		param_groups = utils.get_param_groups(self.model)
		if self.cfg.optimizer.type == 'adamw':
			self.optimizer = torch.optim.AdamW(param_groups)  # to use with ViTs
		# for mixed precision training
		self.fp16_scaler = None
		if self.cfg.meta.use_fp16:
			self.fp16_scaler = torch.cuda.amp.GradScaler()
		
		"""*****init schedulers*****"""
		self.lr_schedule = utils.cosine_scheduler(
			base_value=self.cfg.optimizer.base_lr * (self.cfg.optimizer.batch_size_per_gpu * self.cfg.world_size / 256.),  # linear scaling rule
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
	

	def train_one_epoch(self, epoch):
		
		metric_logger = utils.MetricLogger(delimiter=" ")
		header = f'Epoch: [{epoch}/{self.cfg.optimizer.epochs}]'
		
		# knn mAP metric
		track_knn = self.cfg.knn.track_knn and (epoch % self.cfg.knn.track_knn_it == 0)

		
		end = time.time()
		for iteration, ((y1, y2), labels) in enumerate(metric_logger.log_every(self.data_loader, self.cfg.checkpoint.print_it, header)):
			# measure data loading time
			metric_logger.update(data_time=(time.time()-end))

			# update weight decay and learning rate according to their schedule 
			iteration = len(self.data_loader) * epoch + iteration  # global training iteration
			
			for i, param_group in enumerate(self.optimizer.param_groups):
				param_group["lr"] = self.lr_schedule[iteration]
				if i == 0:  # only the first group is regularized
					param_group["weight_decay"] = self.wd_schedule[iteration]
			
			# move to gpu
			y1 = y1.cuda(non_blocking=True)
			y2 = y2.cuda(non_blocking=True)

			tflag = time.time()
			# forward passes + compute barlow twins loss
			with torch.cuda.amp.autocast(enabled=(self.fp16_scaler is not None)):
				loss = self.model(y1, y2)
			metric_logger.update(forward_time=time.time()-tflag)
			
			if not math.isfinite(loss.item()):
				print(f"Loss is {loss.item()}, stopping training")
				sys.exit(1)
			
			tflag = time.time()
			# gradient update
			self.optimizer.zero_grad()
			if self.fp16_scaler is None:
				loss.backward()
				self.optimizer.step()
			else:
				self.fp16_scaler.scale(loss).backward()
				self.fp16_scaler.step(self.optimizer)
				self.fp16_scaler.update()
			metric_logger.update(backward_time=(time.time()-tflag))

			# logging 
			# if self.cfg.meta.distributed:
			torch.cuda.synchronize()
			loss_val = loss.item()
			lr = self.optimizer.param_groups[0]["lr"]
			wd = self.optimizer.param_groups[0]["weight_decay"]
			metric_logger.update(loss=loss_val)
			metric_logger.update(lr=lr)
			metric_logger.update(wd=wd)


			if self.wandb_run is not None:
				self.wandb_run.log({
					'train_loss': metric_logger.meters['loss'].avg,
					'lr': lr,
					'wd': wd,
					'data_time' : metric_logger.meters['data_time'].avg,
					'forward_time' : metric_logger.meters['forward_time'].avg,
					'backward_time' : metric_logger.meters['backward_time'].avg,
				})
				
			end = time.time()

		# if self.cfg.meta.distributed:
		# gather the stats from all processes
		metric_logger.synchronize_between_processes()
		print(f'Averaged stats: {metric_logger}')
		utils.log_on_master(self.logger, f'Averaged stats: {metric_logger}')

		# return training stats
		train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

		# knn mAP metric
		if track_knn:

			print('Calculating knn mAP')
			utils.log_on_master(self.logger, 'Calculating knn mAP')
			# obtain data loaders
			if self.cfg.data.dataloader.npy:
				knn_train_loader = SpectrogramLoader(self.cfg, pretrain=False, balanced_only=True,test=False).get_loader(drop_last=False)
				knn_test_loader = SpectrogramLoader(self.cfg, pretrain=False, test=True).get_loader(drop_last=False)
			else:
				knn_train_loader = AudioSetLoader(self.cfg, pretrain=False, balanced_only=True,test=False).get_loader(drop_last=False)
				knn_test_loader = AudioSetLoader(self.cfg, pretrain=False, test=True).get_loader(drop_last=False)
			# extract features + calculate knn mAP
			knn_mAP = knn_metric.predict_knn(self.cfg, self.model.module.backbone, knn_train_loader, knn_test_loader)

			print(f'knn mAP: {knn_mAP}')
			utils.log_on_master(self.logger, f'knn mAP: {knn_mAP}')
			train_stats.update({'knn_mAP': knn_mAP})

			if self.wandb_run is not None:
				self.wandb_run.log({
					'test_knn_mAP': knn_mAP,
				})
		
		# save epoch logs
		log_stats = {
			**{f'train_{k}': v for k, v in train_stats.items()},
			'epoch': epoch,
		}
		if utils.is_main_process():
			with (Path(f'{self.cfg.logging.log_dir}/log.txt')).open("a") as f:
				f.write(json.dumps(log_stats) + "\n")
		
		# save checkpoint
		if (epoch % self.cfg.checkpoint.save_epoch_it == 0) or (epoch == self.cfg.optimizer.epochs - 1):
			self.save_checkpoint(epoch, train_stats)


	
	def save_checkpoint(self, epoch, train_stats):
		save_dict = {
			'model': self.model.state_dict(),
			'opt': self.optimizer.state_dict(),
			'epoch': epoch,
			'config': self.cfg,
		}
		if self.fp16_scaler is not None:
			save_dict['fp16_scaler'] = self.fp16_scaler.state_dict()

		utils.save_on_master(save_dict, self.ckpt_path.format(f'epoch-{epoch}'))
	

class BarlowTwins(nn.Module):
	
	def __init__(self, cfg, backbone, projection_sizes, lambd, mask_ratio):
		
		super().__init__()
		self.cfg = cfg
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
		latent1, _, _ = self.backbone(y1, mask_ratio=0.)
		latent2, _, _ = self.backbone(y2, mask_ratio=self.mask_ratio)
		if self.cfg.model.encoder.latent == 'cls':
			# return cls token as global clip representation
			latent1 = latent1[:, 0]
			latent2 = latent2[:, 0]
		else:
			# return mean pool over patch embeddings as global clip representation
			latent1 = torch.mean(latent1[:, 1:], dim=1)
			latent2 = torch.mean(latent2[:, 1:], dim=1)

		z1 = self.projector(latent1)
		z2 = self.projector(latent2)
		
		# empirical cross-correlation matrix
		c = self.bn(z1).T @ self.bn(z2)
		
		# sum the cross-correlation matrix between all gpus
		c.div_(z1.shape[0])
		# if self.cfg.meta.distributed:
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