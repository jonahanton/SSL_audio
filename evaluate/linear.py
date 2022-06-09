"""
Linear evaluation of pre-trained model on AudioSet-20K.
References:
    https://github.com/facebookresearch/msn/blob/main/linear_eval.py
    https://github.com/facebookresearch/dino/blob/main/eval_linear.py
    https://github.com/nttcslab/byol-a
    https://github.com/daisukelab/general-learning/blob/master/MLP/torch_mlp_clf.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn

from sklearn.metrics import average_precision_score

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

        
    
        """*****load pre-trained weights*****"""
        sd = torch.load(cfg.weight_file, map_location='cpu')
        sd = sd['model']
        # remove `module.` prefix
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # remove `backbone.` prefix induced by BarlowTwins wrapper
        sd = {k.replace("backbone.", ""): v for k, v in sd.items()}

        # get patch size
        self.cfg.model.encoder.ps = sd['patch_embed.proj.weight'].shape[2:] 
        # get embedding dim 
        embed_dim = sd['pos_embed'].shape[2]

        """*****build encoder*****"""
        self.cfg.model.encoder.size = 'tiny' if embed_dim == 192 else 'small' if embed_dim == 384 else 'base'
        self.encoder = get_mst_model(
            size=cfg.model.encoder.size,
            patch_size=(cfg.model.encoder.ps[0], cfg.model.encoder.ps[1]),
        )
        # load in weights
        self.encoder.load_state_dict(sd, strict=False)
        print(f'Loaded in pre-trained weights from file {cfg.weight_file}')
        del sd
        # move encoder to gpu
		self.encoder = self.encoder.cuda(self.cfg.gpu)
        self.encoder.eval()

        """*****build linear classifier*****"""
        self.linear_classifier = LinearClassifier(
            dim=embed_dim,
            num_classes=self.cfg.num_classes,
            normalize=self.cfg.optimizer.normalize,
        )
        self.linear_classifier = self.linear_classifier.cuda(self.cfg.gpu)
        if self.cfg.meta.distributed:
            # wrap linear classifier with ddp
			self.linear_classifier = nn.parallel.DistributedDataParallel(self.linear_classifier, device_ids=[self.cfg.gpu])

        """*****prepare optimizer*****"""
        if self.cfg.optimizer.type == 'sgd':
            self.optimizer = torch.optim.SGD(
                params=self.linear_classifier.parameters(),
                lr=self.cfg.optimizer.lr*(self.cfg.optimizer.batch_size_per_gpu*self.cfg.world_size/256.),  # linear scaling rule
                momentum=0.9,
                weight_decay=0,
                nesterov=True,
            )
        # for mixed precision training
		self.fp16_scaler = None
		if self.cfg.meta.use_fp16:
			self.fp16_scaler = torch.cuda.amp.GradScaler()
        
        """*****init schedulers*****"""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cfg.optimizer.epochs, eta_min=0)
        
        """*****init loss*****"""
        self.criterion = nn.BCEWithLogitsLoss()

    
    def train_one_epoch(self, epochs):
        
        self.linear_classifier.train()

        metric_logger = utils.MetricLogger(delimiter=" ")
		header = f'Epoch: [{epoch}/{self.cfg.optimizer.epochs}]'
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

		end = time.time()
        for (inputs, labels) in metric_logger.log_every(self.data_loader_train, self.cfg.checkpoint.print_it, header):
            
            # measure data loading time
			metric_logger.update(data_time=(time.time()-end))

            # move to gpu
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            tflag = time.time()
			# forward passes + compute loss
            with torch.cuda.amp.autocast(enabled=(self.fp16_scaler is not None)):
                with torch.no_grad():
                    outputs = self.encoder(inputs)
			outputs = self.linear_classifier(outputs)
            loss = self.criterion(outputs, labels)
			metric_logger.update(forward_time=time.time()-tflag)
			
			if not math.isfinite(loss.item()):
				print(f"Loss is {loss.item()}, stopping training")
				sys.exit(1)

            # calculate mAP score per batch (to track during training)
            mAP = average_precision_score(labels.cpu().numpy(), outputs.sigmoid().detach().cpu().numpy())
            metric_logger.update(mAP=mAP)

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
			if self.cfg.meta.distributed:
				torch.cuda.synchronize()
			loss_val = loss.item()
			lr = self.optimizer.param_groups[0]["lr"]
			metric_logger.update(loss=loss_val)
			metric_logger.update(lr=lr)

			if self.wandb_run is not None:
				self.wandb_run.log({
					'train_loss': metric_logger.meters['loss'].avg,
					'lr': lr,
                    'train_mAP': metric_logger.meters['mAP'].avg,
					'data_time' : metric_logger.meters['data_time'].avg,
					'forward_time' : metric_logger.meters['forward_time'].avg,
					'backward_time' : metric_logger.meters['backward_time'].avg,
				})

			end = time.time()

        self.scheduler.step()

		if self.cfg.meta.distributed:
			# gather the stats from all processes
			metric_logger.synchronize_between_processes()
		print("Averaged stats:", metric_logger)

		# return training stats
		train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # evaluate on test set (AudioSet eval segments)
        if epoch % self.cfg.val_freq == 0 or epoch == self.cfg.optimizer.epochs - 1:
            test_stats = self.validate()
            test_mAP = test_stats['mAP']
            if self.cfg.meta.distributed:
                test_mAP = utils.AllReduce.apply(test_mAP)
            print(f"Test mAP: {test_mAP:.3f}")
            
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                'test mAP': test_mAP,
            }

            if self.cfg.meta.distributed:
                if utils.is_main_process():
                    with (Path(f'{self.cfg.logging.log_dir}/log.txt')).open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")
            else:
                with (Path(f'{self.cfg.logging.log_dir}/log.txt')).open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")


    @torch.no_grad()
    def validate(self):
        self.linear_classifier.eval()

        stats = {}
        all_targets, all_preds = [], []
        val_loss = 0
        for (inputs, labels) in self.data_loader_test:
            
            all_targets.extend(labels.numpy())
            # move to gpu
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # forward passes + compute loss
            with torch.cuda.amp.autocast(enabled=(self.fp16_scaler is not None)):
                with torch.no_grad():
                    outputs = self.encoder(inputs)
			outputs = self.linear_classifier(outputs)
            loss = self.criterion(outputs, labels)
            val_loss += loss.item()

            all_preds.extend(outputs.sigmoid().detach().cpu().numpy())
        
        val_loss /= len(self.data_loader_test)
        stats['loss'] = val_loss
        stats['mAP'] = average_precision_score(np.array(all_targets), np.array(all_preds)) 
        
        return stats


class LinearClassifier(nn.Module):

    def __init__(self, dim, num_classes=527, normalize=True):
        super().__init__()

        self.normalize = normalize
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.norm(x)
        if self.normalize:  # normalize inputs for linear classifier 
            x = F.normalize(x)
        return self.linear(x)



if __name__ == "__main__":
    pass