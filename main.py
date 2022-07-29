import argparse
import os
from tqdm import tqdm
import time
import datetime
import wandb
import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from augmentations import RunningNorm, NormalizeBatch
from utils import utils, transforms
from utils.torch_mlp_clf import TorchMLPClassifier
import datasets
from model import BarlowTwins


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


def train_one_epoch(args, epoch, model, data_loader, optimizer, fp16_scaler, wandb_run):
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

		with torch.cuda.amp.autocast(enabled=(fp16_scaler is not None)):
			loss = model(pos_1, pos_2)
		forward_time = time.time() - tflag 
		tflag = time.time()

		if not math.isfinite(loss.item()):
			print(f'Loss is {loss.item()}. Stopping training')
			sys.exit(1)

		optimizer.zero_grad()
		if fp16_scaler is None:
			loss.backward()
			optimizer.step()
		else:
			fp16_scaler.scale(loss).backward()
			fp16_scaler.step(optimizer)
			fp16_scaler.update()
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


@torch.no_grad()
def get_embeddings(model, data_loader):
	model.eval()
	embs, targets = [], []
	for data, target in data_loader:
		emb = model(data.cuda(non_blocking=True)).detach().cpu().numpy()
		embs.extend(emb)
		targets.extend(target.numpy())

	return np.array(embs), np.array(targets)


def eval_linear(model, train_loader, val_loader, test_loader):

	print('Extracting embeddings')
	start = time.time()
	X_train, y_train = get_embeddings(model, train_loader)
	X_val, y_val = get_embeddings(model, val_loader)
	X_test, y_test = get_embeddings(model, test_loader)
	print(f'Done\tTime elapsed = {time.time() - start:.2f}s')

	print('Fitting linear classifier')
	start = time.time()
	clf = TorchMLPClassifier(
		hidden_layer_sizes=(1024,),
		max_iter=500,
		early_stopping=True,
		n_iter_no_change=20,
		debug=True,
	)
	clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

	score = clf.score(X_test, y_test)
	print(f'Done\tTime elapsed = {time.time() - start:.2f}s')

	return score


def get_fsd50k(args):
	norm_stats = [-4.950, 5.855]
	eval_train_loader = DataLoader(
		datasets.FSD50K(args, split='train', transform=None, norm_stats=norm_stats),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	eval_val_loader = DataLoader(
		datasets.FSD50K(args, split='val', transform=None, norm_stats=norm_stats),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	eval_test_loader = DataLoader(
		datasets.FSD50K(args, split='test', transform=None, norm_stats=norm_stats),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	return eval_train_loader, eval_val_loader, eval_test_loader



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train barlow twins')
	parser.add_argument('--dataset', default='fsd50k', type=str, choices=DATASETS)
	parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
	parser.add_argument('--epochs', default=100, type=int, help='Number of iterations over the dataset to train for')
	parser.add_argument('--epoch_save_f', default=20, type=int)
	parser.add_argument('--optimizer', default='LARS', type=str, choices = ['Adam', 'AdamW', 'SGD', 'LARS'])
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--wd', type=float, default=1e-5)
	# for LARS optimizer 
	parser.add_argument('--lr_weights', type=float, default=0.4)
	parser.add_argument('--lr_biases', type=float, default=0.0048)
	# model type 
	parser.add_argument('--model_type', default='audiontt', type=str, choices=MODELS)
	parser.add_argument('--squeeze_excitation', action='store_true', default=False)
	# for barlow twins
	parser.add_argument('--alpha', default=1, type=float)
	parser.add_argument('--lmbda', default=0.005, type=float, help='Lambda that controls the on- and off-diagonal terms')
	parser.add_argument('--projector_out_dim', default=8192, type=int)
	parser.add_argument('--projector_n_hidden_layers', default=2, type=int)
	parser.add_argument('--projector_hidden_dim', default=8192, type=int)
	parser.add_argument('--HSIC', action='store_true', default=False)
	# for audio processing
	parser.add_argument('--unit_sec', type=float, default=0.95)
	parser.add_argument('--crop_frames', type=int, default=96)
	parser.add_argument('--sample_rate', type=int, default=16000)
	parser.add_argument('--n_fft', type=int, default=1024)
	parser.add_argument('--win_length', type=int, default=1024)
	parser.add_argument('--hop_length', type=int, default=160)
	parser.add_argument('--n_mels', type=int, default=64)
	parser.add_argument('--f_min', type=int, default=60)
	parser.add_argument('--f_max', type=int, default=7800)
	# data augmentations
	parser.add_argument('--mixup', action='store_true', default=True)
	parser.add_argument('--no_mixup', action='store_false', dest='mixup')
	parser.add_argument('--RRC', action='store_true', default=True)
	parser.add_argument('--no_RRC', action='store_false', dest='RRC')
	parser.add_argument('--RLF', action='store_true', default=True)
	parser.add_argument('--no_RLF', action='store_false', dest='RLF')
	parser.add_argument('--Gnoise', action='store_true', default=False)
	parser.add_argument('--pre_norm', action='store_true', default=False)
	parser.add_argument('--post_norm', action='store_true', default=False)
	# load pre-computed lms 
	parser.add_argument('--load_lms', action='store_true', default=True)
	parser.add_argument('--load_wav', action='store_false', dest='load_lms')
	# distributed training 
	parser.add_argument('--distributed', action='store_true', default=False)
	# data loader
	parser.add_argument('--num_workers', type=int, default=20)
	parser.add_argument('--name', type=str, default=None)
	# mixed precision training
	parser.add_argument('--use_fp16', action='store_true', default=False)
	# masking 
	parser.add_argument('--mask', action='store_true', default=False)
	parser.add_argument('--mask_ratio', type=float, default=0)
	
	# args parse
	args = parser.parse_args()

	# distributed training 
	utils.init_distributed_mode(args)
	args.batch_size_per_gpu = int(args.batch_size / args.world_size)

	# wandb init
	timestamp = datetime.datetime.now().strftime('%H:%M_%h%d')
	save_name = '{}_{}_epochs'.format(args.model_type, args.epochs) if args.name is None else args.name
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
		
	# data prepare
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

	# model setup 
	model = BarlowTwins(args).cuda()

	if args.distributed:
		# sync batch norms
		model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
		# wrap model with ddp
		model = nn.parallel.DistributedDataParallel(
			model,
			device_ids=[args.gpu],
			output_device=args.gpu,
			)
		model_without_ddp = model.module
	else:
		model_without_ddp = model
	
	# optimizer
	if args.optimizer == 'Adam':
		optimizer = optim.Adam(utils.get_param_groups(model), lr=args.lr, weight_decay=args.wd)
	elif args.optimizer == 'AdamW':
		optimizer = optim.AdamW(utils.get_param_groups(model), lr=args.lr, weight_decay=args.wd)
	elif args.optimizer == 'SGD':
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

	# mixed precision
	fp16_scaler = None 
	if args.use_fp16:
		fp16_scaler = torch.cuda.amp.GradScaler()
	

	# model checkpoint path
	ckpt_path = f'results/{args.dataset}/{args.model_type}_{save_name}'
	os.makedirs(ckpt_path, exist_ok=True)

	for epoch in range(1, args.epochs+1):
		train_loss = train_one_epoch(args, epoch, model, train_loader, optimizer, fp16_scaler, wandb_run)
		if epoch % args.epoch_save_f == 0 or epoch == args.epochs:
			utils.save_on_master(model_without_ddp.encoder.state_dict(), ckpt_path + f'/model_{epoch}.pth')
	
	# linear evaluation on fsd50k
	if utils.is_main_process():
		eval_train_loader, eval_val_loader, eval_test_loader = get_fsd50k(args)
		score = eval_linear(model_without_ddp.encoder, eval_train_loader, eval_val_loader, eval_test_loader)
		print(f'Score: {score}')
		wandb_run.log({'FSD50K score': score})

