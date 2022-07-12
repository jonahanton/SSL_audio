"""
End-to-end fine-tuning evaluation of pre-trained model on AudioSet-20K.
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
import torch.distributed as dist

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
from utils.stats import calculate_stats
from data_manager.audioset import AudioSetLoader
from data_manager.audioset_lms import SpectrogramLoader
from models import mae 

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # for sklearn UserWarning 

class FinetuneTrainer:
    
    def __init__(self, cfg, wandb_run, logger):
        
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.logger = logger
        self.construct_model()


    def construct_model(self):
        
        """*****data loaders*****"""
        print(f'Loading AudioSet')
        utils.log_on_master(self.logger, f'Loading AudioSet')
        if self.cfg.data.dataloader.npy:
            # Load in pre-converted raw waveforms (.wav) -> lms (.npy) files 
            self.data_loader_train = SpectrogramLoader(
                self.cfg,
                pretrain=False,
                finetune=True,
                balanced_only=self.cfg.data.audioset.balanced_only,
            ).get_loader()
        else:
            # Load in raw waveforms (.wav)
            self.data_loader_train = AudioSetLoader(
                self.cfg,
                pretrain=False,
                finetune=True,
                balanced_only=self.cfg.data.audioset.balanced_only,
            ).get_loader() 
        print(f'Loaded AudioSet, with {len(self.data_loader_train.dataset)} data points')
        utils.log_on_master(self.logger, f'Loaded AudioSet, with {len(self.data_loader_train.dataset)} data points')

        print(f'Loading AudioSet evaluation set')
        utils.log_on_master(self.logger, f'Loading AudioSet evaluation set')
        if self.cfg.data.dataloader.npy:
            # Load in pre-converted raw waveforms (.wav) -> lms (.npy) files 
            self.data_loader_test = SpectrogramLoader(
                self.cfg,
                pretrain=False,
                test=True,
            ).get_loader()
        else:
            # Load in raw waveforms (.wav)
            self.data_loader_test = AudioSetLoader(
                self.cfg,
                pretrain=False,
                test=True,
            ).get_loader() 
        print(f'Loaded AudioSet evaluation set, with {len(self.data_loader_test.dataset)} data points')
        utils.log_on_master(self.logger, f'Loaded AudioSet evaluation set, with {len(self.data_loader_test.dataset)} data points')
        
    
        """*****load pre-trained weights*****"""
        sd = torch.load(self.cfg.weight_file, map_location='cpu')
        sd = sd['model']
        # remove `module.` prefix
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # remove `backbone.` prefix induced by BarlowTwins wrapper
        sd = {k.replace("backbone.", ""): v for k, v in sd.items()}

        # get patch size
        encoder_ps = sd['patch_embed.proj.weight'].shape[2:] 
        # get embedding dim 
        embed_dim = sd['pos_embed'].shape[2]

        """*****build encoder*****"""
        self.encoder = mae.mae_vit_base_patchX(patch_size=(encoder_ps[0], encoder_ps[1]))
        # load in weights
        self.encoder.load_state_dict(sd, strict=False)
        print(f'Loaded in pre-trained weights from file {self.cfg.weight_file}')
        utils.log_on_master(self.logger, f'Loaded in pre-trained weights from file {self.cfg.weight_file}')
        del sd

        # move encoder to gpu
        self.encoder = self.encoder.cuda(self.cfg.gpu)
        self.encoder.train()

        """*****build finetune classifier*****"""
        self.finetune_classifier = FinetuneClassifier(
            cfg=self.cfg,
            encoder=self.encoder,
            dim=embed_dim,
            num_classes=self.cfg.num_classes,
            normalize=self.cfg.optimizer.normalize,
        )
        self.finetune_classifier = self.finetune_classifier.cuda(self.cfg.gpu)
        if self.cfg.meta.distributed:
            # synchronize batch norms
            self.finetune_classifier = nn.SyncBatchNorm.convert_sync_batchnorm(self.finetune_classifier)
            # wrap finetune classifier with ddp
            self.finetune_classifier = nn.parallel.DistributedDataParallel(
                self.finetune_classifier,
                device_ids=[self.cfg.gpu],
                output_device=self.cfg.gpu,
            )

        """*****prepare optimizer*****"""
        param_groups = utils.get_param_groups(self.finetune_classifier)
        if self.cfg.optimizer.type == 'sgd':
            self.optimizer = torch.optim.SGD(
                params=param_groups,
                momentum=0.9,
                weight_decay=0,
                nesterov=True,
            )
        elif self.cfg.optimizer.type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                params=param_groups,
            )
        
        # for mixed precision training
        self.fp16_scaler = None
        if self.cfg.meta.use_fp16:
            self.fp16_scaler = torch.cuda.amp.GradScaler()
        
        """*****init schedulers*****"""
        self.lr_schedule = utils.cosine_scheduler(
            base_value=self.cfg.optimizer.base_lr * (self.cfg.optimizer.batch_size_per_gpu * self.cfg.world_size / 256.),  # linear scaling rule
            final_value=self.cfg.optimizer.final_lr,
            epochs=self.cfg.optimizer.epochs, 
            niter_per_ep=len(self.data_loader_train),
            warmup_epochs=self.cfg.optimizer.warmup_epochs,
        )
        if self.cfg.optimizer.final_weight_decay is None:
            self.cfg.optimizer.final_weight_decay = self.cfg.optimizer.weight_decay
        self.wd_schedule = utils.cosine_scheduler(
            base_value=self.cfg.optimizer.weight_decay,
            final_value=self.cfg.optimizer.final_weight_decay,
            epochs=self.cfg.optimizer.epochs,
            niter_per_ep=len(self.data_loader_train),
        )
        
        """*****init loss*****"""
        self.criterion = nn.BCEWithLogitsLoss()

    
    def train_one_epoch(self, epoch):
        
        self.finetune_classifier.train()

        metric_logger = utils.MetricLogger(delimiter=" ")
        header = f'Epoch: [{epoch}/{self.cfg.optimizer.epochs}]'
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        end = time.time()
        for iteration, (inputs, labels) in enumerate(metric_logger.log_every(self.data_loader_train, self.cfg.checkpoint.print_it, header)):
            
            # measure data loading time
            metric_logger.update(data_time=(time.time()-end))

            # update weight decay and learning rate according to their schedule 
            iteration = len(self.data_loader_train) * epoch + iteration  # global training iteration
			
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.lr_schedule[iteration]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.wd_schedule[iteration]

            # move to gpu
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            tflag = time.time()
            # forward passes + compute loss
            with torch.cuda.amp.autocast(enabled=(self.fp16_scaler is not None)):
                outputs = self.finetune_classifier(inputs)
                loss = self.criterion(outputs, labels)
            metric_logger.update(forward_time=time.time()-tflag)

            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training")
                utils.log_on_master(self.logger, f"Loss is {loss.item()}, stopping training")
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

        # evaluate on test set (AudioSet eval segments)
        if epoch % self.cfg.val_freq == 0 or epoch == self.cfg.optimizer.epochs - 1:
            
            print('Starting evaluation on test set')
            utils.log_on_master(self.logger, 'Starting evaluation on test set')
            test_mAP, test_loss = self.validate()
            test_loss = utils.all_reduce_mean(test_loss)
            print(f'Test mAP: {test_mAP}\tTest loss: {test_loss}')
            utils.log_on_master(self.logger, f'Test mAP: {test_mAP}\tTest loss: {test_loss}')

            if self.wandb_run is not None:
                self.wandb_run.log({
                    'test_mAP': test_mAP,
                })
            
            log_stats = {
                'test_mAP': test_mAP,
                'test_loss': test_loss,
                **{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
            }

            if utils.is_main_process():
                with (Path(f'{self.cfg.logging.log_dir}/log.txt')).open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")


    @torch.no_grad()
    def validate(self):
        self.finetune_classifier.eval()

        stats = {}
        all_targets, all_preds = [], []
        val_loss = 0
        for (inputs, labels) in self.data_loader_test:
            
            # move to gpu
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # forward passes + compute loss
            with torch.cuda.amp.autocast(enabled=(self.fp16_scaler is not None)):
                with torch.no_grad():
                    outputs = self.finetune_classifier(inputs)
                loss = self.criterion(outputs, labels)
            val_loss += loss.item()

            all_preds.append(outputs.sigmoid())
            all_targets.append(labels)
        
        val_loss /= len(self.data_loader_test)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        if self.cfg.meta.distributed:
            # gather preds from all processes     
            all_preds_list = [torch.zeros_like(all_preds) for _ in range(self.cfg.world_size)]
            handle = dist.all_gather(all_preds_list, all_preds, async_op=True)
            handle.wait()
            all_preds = torch.cat(all_preds_list, dim=0)
            # gather targets from all processes
            all_targets_list = [torch.zeros_like(all_targets) for _ in range(self.cfg.world_size)]
            handle = dist.all_gather(all_targets_list, all_targets, async_op=True)
            handle.wait()
            all_targets = torch.cat(all_targets_list, dim=0)
        
        # calculate mAP
        mAP = average_precision_score(y_true=all_targets.cpu(), y_score=all_preds.cpu(), average='macro') 
        return mAP, val_loss


class FinetuneClassifier(nn.Module):

    def __init__(self, cfg, encoder, dim, num_classes=527, normalize=True):
        super().__init__()

        self.cfg = cfg
        self.encoder = encoder
        self.normalize = normalize
        # self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    
    def forward(self, x):
        x, _, _ = self.encoder(x, mask_ratio=0.)
        if self.cfg.model.encoder.latent == 'cls':
            # return cls token as global clip representation
            x = x[:, 0]
        else:
            # return mean pool over patch embeddings as global clip representation
            x = torch.mean(x[:, 1:], dim=1)
        x = x.contiguous()
        x = x.view(x.size(0), -1)  # flatten
        # x = self.norm(x)
        if self.normalize:  # normalize inputs for linear classifier 
            x = F.normalize(x)
        return self.linear(x)



if __name__ == "__main__":
    pass