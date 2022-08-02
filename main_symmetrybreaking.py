import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime
import wandb

from augmentations import RunningNorm, NormalizeBatch
from utils import utils, transforms, hyperparameters
import datasets
from model import BarlowTwinsBYOL


MODELS = [
	'resnet50', 'resnet50_ReGP_NRF',
	'audiontt',
	'vit_base', 'vit_small', 'vit_tiny',
	'vitc_base', 'vitc_small', 'vitc_tiny'
]

DATASETS = [
	'fsd50k',
	'audioset',
	'librispeech',
	'fsd50k+librispeech',
]

if torch.cuda.is_available():
	torch.backends.cudnn.benchmark = True


def train_one_epoch(args, epoch, model, data_loader, optimizer, wandb_run):
	model.train()
	total_loss, total_num, train_bar = 0, 0, tqdm(data_loader)
	
	total_data_time, total_forward_time, total_backward_time = 0, 0, 0
	tflag = time.time()
	for data_tuple in train_bar:
		data_time = time.time() - tflag
		tflag = time.time()

		(pos_1, pos_2), _ = data_tuple

		if args.post_norm:
			# Post-normalization block from BYOL-A [Niizumi et al., 2021]
			bs = pos_1.shape[0]
			paired_inputs = torch.cat([pos_1, pos_2])  # [(B,1,F,T), (B,1,F,T)] -> (2*B,1,F,T)
			paired_inputs = NormalizeBatch()(paired_inputs)
			pos_1 = paired_inputs[:bs]
			pos_2 = paired_inputs[bs:]

		pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

		loss = model(pos_1, pos_2)
		forward_time = time.time() - tflag 
		tflag = time.time()

		# update target encoder moving average
		model.update_moving_average()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		backward_time = time.time() - tflag 

		total_num += args.batch_size_per_gpu
		total_loss += loss.item() * args.batch_size_per_gpu

		total_data_time += data_time
		total_forward_time += forward_time 
		total_backward_time += backward_time

		train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Data time {:.2f}({:.2f}) Forward time {:.2f}({:.2f}) Backward time {:.2f}({:.2f}))'.format(
								  epoch, args.epochs, total_loss / total_num, 
								  data_time, total_data_time,
								  forward_time, total_forward_time,
								  backward_time, total_backward_time))
		
		if wandb_run is not None:
			wandb_run.log({'Loss': total_loss / total_num})

		tflag = time.time()
		
	return total_loss / total_num


def get_data(args):
	if args.dataset == 'fsd50k':
		# fsd50k [mean, std] (lms)
		norm_stats = [-4.950, 5.855]
		len_files = 40966 
		if args.pre_norm:
			transform = nn.Sequential(
				RunningNorm(epoch_samples=len_files),
				transforms.AudioPairTransform(args, train_transform=True),
			)
			train_data = datasets.FSD50K(args, split='train_val', transform=transform, norm_stats=None)
		else:
			transform = transforms.AudioPairTransform(args, train_transform=True)
			train_data = datasets.FSD50K(args, split='train_val', transform=transform, norm_stats=norm_stats)
	elif args.dataset == 'librispeech':
		# librispeech960 [mean, std] (lms)
		norm_stats = [-3.332, 4.205]
		train_data = datasets.LibriSpeech(args, train=True, transform=transforms.AudioPairTransform(args, train_transform=True), norm_stats=norm_stats)
	elif args.dataset == 'fsd50k+librispeech':
		norm_stats_fsd50k = [-4.950, 5.855]
		norm_stats_librispeech = [-3.332, 4.205]
		train_data = torch.utils.data.dataset.ConcatDataset([
			datasets.FSD50K(args, split='train_val', transform=transforms.AudioPairTransform(args, train_transform=True), norm_stats=norm_stats_fsd50k),
			datasets.LibriSpeech(args, train=True, transform=transforms.AudioPairTransform(args, train_transform=True), norm_stats=norm_stats_librispeech),
		])
	elif args.dataset == 'audioset':
		norm_stats = [-0.8294, 4.6230]
		train_data = datasets.AudioSet(args, transform=transforms.AudioPairTransform(args, train_transform=True), norm_stats=norm_stats)
	
	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
	else:
		train_sampler = None
		
	train_loader = DataLoader(train_data, batch_size=args.batch_size_per_gpu, shuffle=(True if train_sampler is None else False),
							  num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

	return train_loader


def get_optimizer(args):

	if args.optimizer == 'Adam':
		args.wd = 0
		optimizer = optim.Adam(utils.get_param_groups(model), lr=args.lr, weight_decay=args.wd)
	elif args.optimizer == 'AdamW':
		optimizer = optim.AdamW(utils.get_param_groups(model), lr=args.lr, weight_decay=args.wd)
	elif args.optimizer == 'SGD':
		args.wd = 0
		optimizer = optim.SGD(utils.get_param_groups(model), lr=args.lr, weight_decay=args.wd)
	elif args.optimizer == 'LARS':
		# separate lr for weights and biases using LARS optimizer
		param_weights = []
		param_biases = []
		for param in model.parameters():
			if param.ndim == 1:
				param_biases.append(param)
			else:
				param_weights.append(param)
		parameters = [
			{'params': param_weights, 'lr': args.lr_weights},
			{'params': param_biases, 'lr': args.lr_biases},
		]
		optimizer = utils.LARS(parameters, lr=0, weight_decay=args.wd,
			weight_decay_filter=True, lars_adaptation_filter=True)
	
	return optimizer

if __name__ == '__main__':
	parser_a = argparse.ArgumentParser(description='Model args')
	parser_a.add_argument('--model_type', default='audiontt', type=str, choices=MODELS)
	model_args = parser_a.parse_args()
	parser_b = argparse.ArgumentParser(description='All args', parents=hyperparameters.get_hyperparameters(model_args))
	parser_b.add_argument('--stop_gradient', action='store_true', default=True)
	parser_b.add_argument('--no_stop_gradient', action='store_false', dest='stop_gradient')
	parser_b.add_argument('--predictor', action='store_true', default=True)
	parser_b.add_argument('--no_predictor', action='store_false', dest='predictor')
	parser_b.add_argument('--moving_average_decay', type=float, default=0.99)
	args = parser_b.parse_args()
	args.model_type = model_args.model_type

	# distributed training 
	utils.init_distributed_mode(args)
	args.batch_size_per_gpu = int(args.batch_size / args.world_size)

	# wandb init
	timestamp = datetime.datetime.now().strftime('%H:%M_%h%d')
	save_name = '(asymm)_{}_{}_epochs'.format(args.model_type, args.epochs) if args.name == '' else args.name
	save_name += timestamp
	if utils.is_main_process():
		wandb_run = wandb.init(
				project='Pre-training {}'.format(args.dataset),
				config=args,
				settings=wandb.Settings(start_method="fork"),
				name=save_name,
			)
	else:
		wandb_run = None
		
	# data 
	train_loader = get_data(args)
	# model 
	model = BarlowTwinsBYOL(args).cuda()

	if args.distributed:
		model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
		model = nn.parallel.DistributedDataParallel(
			model,
			device_ids=[args.gpu],
			output_device=args.gpu,
			)
		model_without_ddp = model.module
	else:
		model_without_ddp = model

	# optimizer
	optimizer = get_optimizer(args)

	# model checkpoint path
	ckpt_path = f'results/{args.dataset}/{args.model_type}_{save_name}'
	os.makedirs(ckpt_path, exist_ok=True)

	# training 
	for epoch in range(1, args.epochs+1):
		train_loss = train_one_epoch(args, epoch, model, train_loader, optimizer, wandb_run)
		if epoch % args.epoch_save_f == 0 or epoch == args.epochs:
			utils.save_on_master(model_without_ddp.encoder.state_dict(), ckpt_path + f'/model_{epoch}.pth')
