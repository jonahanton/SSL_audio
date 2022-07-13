"""
Implementation of DINO [Caron et al., 2021], 
adapted from
	https://github.com/facebookresearch/dino
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
from models import mae 


class DINOTrainer:

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

		"""*****build models (student + teacher)*****"""
		if self.cfg.model.encoder.type == 'transformer':
			self.student = mae.mae_vit_base_patchX(patch_size=self.cfg.model.encoder.ps, drop_path_rate=self.cfg.model.drop_path_rate)
			self.teacher = mae.mae_vit_base_patchX(patch_size=self.cfg.model.encoder.ps)
			embed_dim = self.student.embed_dim
		
			
		# wrap in DINO head
		self.student = DINOHead(in_dim=embed_dim, out_dim=self.cfg.model.projection.out_dim)
		self.teacher = DINOHead(in_dim=embed_dim, out_dim=self.cfg.model.projection.out_dim)

		# teacher and student start with the same weights
		self.teacher.load_state_dict(self.student.state_dict())
		
		# move to gpu
		self.student = self.student.cuda(self.cfg.gpu)
		self.teacher = self.teacher.cuda(self.cfg.gpu)
		
		if self.cfg.meta.distributed:
			# synchronize batch norms
			self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
			self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)
			# wrap with ddp
			self.student = nn.parallel.DistributedDataParallel(
				self.student,
				device_ids=[self.cfg.gpu],
				output_device=self.cfg.gpu,
			)
			self.student_without_ddp = self.student.module
			self.teacher = nn.parallel.DistributedDataParallel(
				self.teacher,
				device_ids=[self.cfg.gpu],
				output_device=self.cfg.gpu,
			)
			self.teacher_without_ddp = self.teacher.module

		# there is no backpropagation through the teacher, so no need for gradients
		for p in self.teacher.parameters():
			p.requires_grad = False

		"""*****prepare loss*****"""
		self.dino_loss = DINOLoss(
			out_dim=self.cfg.model.projection.out_dim,
			ncrops=2,
			warmup_teacher_temp=self.cfg.model.warmup_teacher_temp,
			teacher_temp=self.cfg.model.teacher_temp,
			warmup_teacher_temp_epochs=self.cfg.model.warmup_teacher_temp_epochs,
			nepochs=self.cfg.optimizer.epochs,
		).cuda(self.cfg.gpu)

		"""*****prepare optimizer*****"""
		param_groups = utils.get_param_groups(self.student)
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
		 # momentum parameter is increased to 1. during training with a cosine schedule
		self.momentum_schedule = utils.cosine_scheduler(
			base_value=self.cfg.model.momentum_teacher,
			final_value=1,
			epochs=self.cfg.optimizer.epochs,
			niter_per_ep=len(self.data_loader),
		)
	

	def train_one_epoch(self, epoch):
		
		metric_logger = utils.MetricLogger(delimiter=" ")
		header = f'Epoch: [{epoch}/{self.cfg.optimizer.epochs}]'
		
		# knn mAP metric
		track_knn = self.cfg.knn.track_knn and (epoch % self.cfg.knn.track_knn_it == 0)

		
		end = time.time()
		for iteration, (images, labels) in enumerate(metric_logger.log_every(self.data_loader, self.cfg.checkpoint.print_it, header)):
			# measure data loading time
			metric_logger.update(data_time=(time.time()-end))

			# update weight decay and learning rate according to their schedule 
			iteration = len(self.data_loader) * epoch + iteration  # global training iteration
			
			for i, param_group in enumerate(self.optimizer.param_groups):
				param_group["lr"] = self.lr_schedule[iteration]
				if i == 0:  # only the first group is regularized
					param_group["weight_decay"] = self.wd_schedule[iteration]
			
			# move to gpu
			images = [im.cuda(non_blocking=True) for im in images]

			tflag = time.time()
			# forward passes + compute barlow twins loss
			with torch.cuda.amp.autocast(enabled=(self.fp16_scaler is not None)):
				teacher_output = self.teacher(images[:2])  # only the 2 global views pass through the teacher
				student_output = self.student(images)
				loss = self.dino_loss(student_output, teacher_output, epoch)
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

			# EMA update for the teacher
			with torch.no_grad():
				m = self.momentum_schedule[iteration]  # momentum parameter
				for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
					param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

			# logging 
			if self.cfg.meta.distributed:
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
			'model': self.student.state_dict(),
			'opt': self.optimizer.state_dict(),
			'epoch': epoch,
			'config': self.cfg,
		}
		if self.fp16_scaler is not None:
			save_dict['fp16_scaler'] = self.fp16_scaler.state_dict()

		utils.save_on_master(save_dict, self.ckpt_path.format(f'epoch-{epoch}'))
	

class DINOHead(nn.Module):
	def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
		super().__init__()
		nlayers = max(nlayers, 1)
		if nlayers == 1:
			self.mlp = nn.Linear(in_dim, bottleneck_dim)
		else:
			layers = [nn.Linear(in_dim, hidden_dim)]
			if use_bn:
				layers.append(nn.BatchNorm1d(hidden_dim))
			layers.append(nn.GELU())
			for _ in range(nlayers - 2):
				layers.append(nn.Linear(hidden_dim, hidden_dim))
				if use_bn:
					layers.append(nn.BatchNorm1d(hidden_dim))
				layers.append(nn.GELU())
			layers.append(nn.Linear(hidden_dim, bottleneck_dim))
			self.mlp = nn.Sequential(*layers)
		self.apply(self._init_weights)
		self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
		self.last_layer.weight_g.data.fill_(1)
		if norm_last_layer:
			self.last_layer.weight_g.requires_grad = False

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.mlp(x)
		x = nn.functional.normalize(x, dim=-1, p=2)
		x = self.last_layer(x)
		return x


class DINOLoss(nn.Module):
	def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
				 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
				 center_momentum=0.9):
		super().__init__()
		self.student_temp = student_temp
		self.center_momentum = center_momentum
		self.ncrops = ncrops
		self.register_buffer("center", torch.zeros(1, out_dim))
		# we apply a warm up for the teacher temperature because
		# a too high temperature makes the training instable at the beginning
		self.teacher_temp_schedule = np.concatenate((
			np.linspace(warmup_teacher_temp,
						teacher_temp, warmup_teacher_temp_epochs),
			np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
		))

	def forward(self, student_output, teacher_output, epoch):
		"""
		Cross-entropy between softmax outputs of the teacher and student networks.
		"""
		student_out = student_output / self.student_temp
		student_out = student_out.chunk(self.ncrops)

		# teacher centering and sharpening
		temp = self.teacher_temp_schedule[epoch]
		teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
		teacher_out = teacher_out.detach().chunk(2)

		total_loss = 0
		n_loss_terms = 0
		for iq, q in enumerate(teacher_out):
			for v in range(len(student_out)):
				if v == iq:
					# we skip cases where student and teacher operate on the same view
					continue
				loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
				total_loss += loss.mean()
				n_loss_terms += 1
		total_loss /= n_loss_terms
		self.update_center(teacher_output)
		return total_loss

	@torch.no_grad()
	def update_center(self, teacher_output):
		"""
		Update center used for teacher output.
		"""
		batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
		if utils.is_dist_avail_and_initialized():
			dist.all_reduce(batch_center)
		batch_center = batch_center / (len(teacher_output) * utils.get_world_size())

		# ema update
		self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)