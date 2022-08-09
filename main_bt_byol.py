import argparse
import os
from tqdm import tqdm
import time
import datetime
import wandb
import math
import random
import sys
import logging
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import numpy as np

from augmentations import RunningNorm, NormalizeBatch
from utils.loss import BarlowTwinsLoss
from utils import utils, transforms, hyperparameters
from utils.torch_mlp_clf import TorchMLPClassifier
import datasets
from model import ModelWrapper, BarlowTwinsHead, BarlowTwinsPredictor

off_diagonal = utils.off_diagonal

CLASSES = dict(
	fsd50k=200,
	nsynth=88,
	cifar10=10,
)


if torch.cuda.is_available():
	torch.backends.cudnn.benchmark = True


def train_one_epoch(args, epoch, online_encoder, online_encoder_without_ddp, 
					target_encoder, target_encoder_without_ddp, target_ema_updater,
					barlow_twins_loss, data_loader, optimizer, fp16_scaler, logger, 
					wandb_run):
	online_encoder.train()
	total_loss, total_num, train_bar = 0, 0, tqdm(data_loader)
	if args.masked_recon:
		total_bt_loss, total_recon_loss = 0, 0
	
	total_data_time, total_forward_time, total_backward_time = 0, 0, 0
	tflag = time.time()
	for iteration, (images, _) in enumerate(train_bar):
		data_time = time.time() - tflag

		iteration += len(data_loader) * (epoch - 1)  # global training iteration

		tflag = time.time()

		# post-normalization block from BYOL-A [Niizumi et al., 2021]
		if args.post_norm:
			norm_images = []
			for im in images:
				norm_images.append(NormalizeBatch()(im))
			images = norm_images

		# move images to gpu
		images = [im.cuda(non_blocking=True) for im in images]

		# mask ratio
		if args.mask:
			if args.random_mask_ratio:
				# randomly sample r ~ U(0.02, 0.2) with p = 0.5
				mask_ratio = utils.generate_random(l=0.02, h=0.2, p=0.5)
			else:
				mask_ratio = args.mask_ratio
		else:
			mask_ratio = 0

		# forward passes + compute barlow twins loss
		with torch.cuda.amp.autocast(enabled=(fp16_scaler is not None)):
			
			# online encoder
			online_output = online_encoder(
				images[:2],
				mask_ratio=mask_ratio,
				masked_recon=args.masked_recon,
				ncrops=2, # 2 global crops both passed through the online encoder 
			)
			# masked recon
			if args.masked_recon:
				online_output, recon_loss = online_output
			# predictor
			online_output = online_predictor(online_output)

			# target encoder
			target_output = target_encoder(
				images,
				ncrops=args.local_crops_number+2, # 2 global crops + all local crops passed through the target encoder
			)


			bt_loss = barlow_twins_loss(
				online_output,
				target_output,
				ngcrops_each=2,
			)

		forward_time = time.time() - tflag 
		tflag = time.time()

		loss = bt_loss
		if args.masked_recon:
			loss += recon_loss

		if not math.isfinite(loss.item()):
			print(f'Loss is {loss.item()}. Stopping training')
			sys.exit(1)

		# update target encoder moving average
		if args.stop_gradient:
			utils.update_moving_average(
				target_ema_updater,
				target_encoder_without_ddp,
				online_encoder_without_ddp,
			)

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
		if args.masked_recon:
			total_bt_loss += bt_loss.item() * args.batch_size_per_gpu
			total_recon_loss += recon_loss.item() * args.batch_size_per_gpu

		total_data_time += data_time
		total_forward_time += forward_time 
		total_backward_time += backward_time

		train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Data time {:.2f}({:.2f}) Forward time {:.2f}({:.2f}) Backward time {:.2f}({:.2f}))'.format(
								  epoch, args.epochs, total_loss / total_num, 
								  data_time, total_data_time,
								  forward_time, total_forward_time,
								  backward_time, total_backward_time))
		
		if logger is not None:
			logger.info('epoch,{},step,{},loss,{}'.format(
						epoch, iteration, total_loss / total_num))
		if wandb_run is not None:
			to_log = {'Loss': total_loss / total_num}
			if args.masked_recon:
				to_log['barlow twins loss'] = total_bt_loss / total_num
				to_log['masked recon loss'] = total_recon_loss / total_num
			wandb_run.log(to_log)

		tflag = time.time()
		
	return total_loss / total_num


@torch.no_grad()
def get_embeddings(model, data_loader, fp16_scaler):
	model.eval()
	embs, targets = [], []
	for data, target in data_loader:
		with torch.cuda.amp.autocast(enabled=(fp16_scaler is not None)):
			if 'vit' in args.model_type:
				emb = utils.encode_vit(
					model.encoder,
					data.cuda(non_blocking=True),
					split_frames=True,
					use_cls=args.use_cls,
				)
			else:
				emb = model(data.cuda(non_blocking=True))
		if isinstance(emb, list):
			emb = emb[-1]
		emb = emb.detach().cpu().numpy()
		embs.extend(emb)
		targets.extend(target.numpy())

	return np.array(embs), np.array(targets)


def eval_linear(model, train_loader, val_loader, test_loader, use_fp16):

	# mixed precision
	fp16_scaler = None
	if use_fp16:
		fp16_scaler = torch.cuda.amp.GradScaler()

	print('Extracting embeddings')
	start = time.time()
	X_train, y_train = get_embeddings(model, train_loader, fp16_scaler)
	X_val, y_val = get_embeddings(model, val_loader, fp16_scaler)
	X_test, y_test = get_embeddings(model, test_loader, fp16_scaler)
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

	score_all = clf.score(X_test, y_test)
	print(f'Done\tTime elapsed = {time.time() - start:.2f}s')

	# Low-shot linear evaluation
	print('Performing linear evaluation with 5 example per class')
	start = time.time()
	score_5 = utils.eval_linear_low_shot(X_train, y_train, X_val, y_val, X_test, y_test, n=5)
	print(f'Done\tTime elapsed = {time.time() - start:.2f}s')

	results_dict = dict(
		score_all = score_all,
		score_5 = score_5,
	)

	return results_dict


def get_fsd50k(args):
	norm_stats = [-4.950, 5.855]
	eval_train_loader = DataLoader(
		datasets.FSD50K(args, split='train', transform=None, norm_stats=norm_stats, crop_frames=711),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	eval_val_loader = DataLoader(
		datasets.FSD50K(args, split='val', transform=None, norm_stats=norm_stats, crop_frames=711),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	eval_test_loader = DataLoader(
		datasets.FSD50K(args, split='test', transform=None, norm_stats=norm_stats, crop_frames=711),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	return eval_train_loader, eval_val_loader, eval_test_loader


def get_data(args):
	if args.dataset == 'cifar10':
		train_data = torchvision.datasets.CIFAR10(root='data', train=True, transform=transforms.CifarPairTransform(train_transform=True), download=True)
		memory_data = torchvision.datasets.CIFAR10(root='data', train=True, transform=transforms.CifarPairTransform(train_transform=False), download=True)
		test_data = torchvision.datasets.CIFAR10(root='data', train=False, transform=transforms.CifarPairTransform(train_transform=False), download=True)

		train_loader = DataLoader(train_data, batch_size=args.batch_size_per_gpu, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
		memory_loader = DataLoader(memory_data, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=args.num_workers, pin_memory=True)
		test_loader = DataLoader(test_data, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=args.num_workers, pin_memory=True)
		
		return train_loader, memory_loader, test_loader
	elif args.dataset == 'fsd50k':
		# fsd50k [mean, std] (lms)
		norm_stats = [-4.950, 5.855]
		len_files = 40966 
		if args.pre_norm:
			transform = nn.Sequential(
				RunningNorm(epoch_samples=len_files),
				transforms.AudioPairTransform(args),
			)
			train_data = datasets.FSD50K(args, split='train_val', transform=transform, norm_stats=None)
		else:
			transform = transforms.AudioPairTransform(args)
			train_data = datasets.FSD50K(args, split='train_val', transform=transform, norm_stats=norm_stats)
	elif args.dataset == 'librispeech':
		# librispeech960 [mean, std] (lms)
		norm_stats = [-3.332, 4.205]
		train_data = datasets.LibriSpeech(args, train=True, transform=transforms.AudioPairTransform(args), norm_stats=norm_stats)
	elif args.dataset == 'fsd50k+librispeech':
		norm_stats_fsd50k = [-4.950, 5.855]
		norm_stats_librispeech = [-3.332, 4.205]
		train_data = torch.utils.data.dataset.ConcatDataset([
			datasets.FSD50K(args, split='train_val', transform=transforms.AudioPairTransform(args), norm_stats=norm_stats_fsd50k),
			datasets.LibriSpeech(args, train=True, transform=transforms.AudioPairTransform(args), norm_stats=norm_stats_librispeech),
		])
	elif args.dataset == 'audioset':
		norm_stats = [-0.8294, 4.6230]
		train_data = datasets.AudioSet(args, transform=transforms.AudioPairTransform(args), norm_stats=norm_stats)
	
	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
	else:
		train_sampler = None
		
	train_loader = DataLoader(train_data, batch_size=args.batch_size_per_gpu, shuffle=(True if train_sampler is None else False),
							  num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

	return train_loader


def get_optimizer(args, online_encoder_without_ddp, online_predictor_without_ddp, target_encoder_without_ddp):

	params = utils.get_param_groups(online_encoder_without_ddp)
	params.extend(utils.get_param_groups(online_predictor_without_ddp))
	if not args.stop_gradient:
		params.extend(utils.get_param_groups(target_encoder_without_ddp))
	
	if args.optimizer == 'Adam':
		args.wd = 0
		optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
	elif args.optimizer == 'AdamW':
		optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
	elif args.optimizer == 'SGD':
		args.wd = 0
		optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd)
	elif args.optimizer == 'LARS':
		# separate lr for weights and biases using LARS optimizer
		param_weights, param_weights = [], []
		# online encoder params
		for param in online_encoder_without_ddp.parameters():
			if param.ndim == 1:
				online_param_biases.append(param)
			else:
				online_param_weights.append(param)
		# online predictor params
		for param in online_predictor_without_ddp.parameters():
			if param.ndim == 1:
				param_biases.append(param)
			else:
				param_weights.append(param)
		# target encoder
		if not args.stop_gradient:
			for param in target_encoder_without_ddp.parameters():
				if param.ndim == 1:
					target_param_biases.append(param)
				else:
					target_param_weights.append(param)
		parameters = [
			{'params': param_weights, 'lr': args.lr_weights},
			{'params': param_biases, 'lr': args.lr_biases},
		]

		optimizer = utils.LARS(parameters, lr=0, weight_decay=args.wd,
			weight_decay_filter=True, lars_adaptation_filter=True)
	
	return optimizer


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training args', parents=hyperparameters.get_hyperparameters())
	parser.add_argument('--stop_gradient', action='store_true', default=True)
	parser.add_argument('--no_stop_gradient', action='store_false', dest='stop_gradient')
	parser.add_argument('--predictor', action='store_true', default=True)
	parser.add_argument('--no_predictor', action='store_false', dest='predictor')
	parser.add_argument('--moving_average_decay', type=float, default=0.99)
	args = parser.parse_args()
	hyperparameters.setup_hyperparameters(args)


	# distributed training 
	utils.init_distributed_mode(args)
	args.batch_size_per_gpu = int(args.batch_size / args.world_size)

	# wandb init
	timestamp = datetime.datetime.now().strftime('%H:%M_%h%d')
	save_name = '(byol-ish)_{}_{}_epochs'.format(args.model_type, args.epochs) if args.name == '' else '{}_{}'.format(args.model_type, args.name)
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

	# logging
	if utils.is_main_process():
		log_dir = f"logs/training/{args.dataset}/{save_name}/"
		os.makedirs(log_dir, exist_ok=True)
		log_path = os.path.join(log_dir, f"log.csv")
		logger = logging.getLogger()
		logger.setLevel(logging.INFO)  # Setup the root logger
		logger.addHandler(logging.FileHandler(log_path, mode="w"))
	else:
		logger = None
		
	# data 
	if args.dataset == 'cifar10':
		assert args.distributed == False, f'Distributed training is not supported with cifar10'
		train_loader, memory_loader, test_loader = get_data(args)
	else:
		train_loader = get_data(args)

	# online encoder
	online_encoder = ModelWrapper(args)
	# multi-crop wrapper handles forward with inputs of different resolutions
	online_encoder = utils.MultiCropWrapper(
		backbone=online_encoder,
		head=BarlowTwinsHead(
			args,
			in_dim=online_encoder.feature_dim,
		),
	)
	# move network to gpu
	online_encoder = online_encoder.cuda()

	# predictor network
	online_predictor = BarlowTwinsPredictor(
		in_dim=args.projector_out_dim,
		use=args.predictor,
	)
	# move network to gpu
	online_predictor = online_predictor.cuda()

	# target encoder
	target_encoder = ModelWrapper(args)
	# multi-crop wrapper handles forward with inputs of different resolutions
	target_encoder = utils.MultiCropWrapper(
		backbone=target_encoder,
		head=BarlowTwinsHead(
			args,
			in_dim=target_encoder.feature_dim,
		),
	)
	# move network to gpu
	target_encoder = target_encoder.cuda()

	# initialise target encoder weights from online encoder
	target_encoder.load_state_dict(online_encoder.state_dict(), strict=True)
	# stop gradient for target encoder
	if args.stop_gradient:
		for p in target_encoder.parameters():
			p.requires_grad = False
	target_ema_updater = utils.EMA(args.moving_average_decay)

	
	set up model for distributed training
	if args.distributed:
		online_encoder_without_ddp = utils.model_setup_ddp(args.gpu, online_encoder)
		online_predictor_without_ddp = utils.model_setup_ddp(args.gpu, online_predictor)
		target_encoder_without_ddp = utils.model_setup_ddp(args.gpu, target_encoder)
	else:
		online_encoder_without_ddp = online_encoder
		online_predictor_without_ddp = online_predictor
		target_encoder_without_ddp = target_encoder


	# prepare loss
	barlow_twins_loss = BarlowTwinsLoss(
		args,
		ncrops=args.local_crops_number+2,  # total number of crops = 2 global crops + local_crops_number
	).cuda()

	# optimizer
	optimizer = get_optimizer(
		args,
		online_encoder_without_ddp,
		target_encoder_without_ddp,
	)
	
	# mixed precision
	fp16_scaler = None 
	if args.use_fp16:
		fp16_scaler = torch.cuda.amp.GradScaler()	

	# model checkpoint path
	ckpt_path = f'results/{args.dataset}/{save_name}'
	os.makedirs(ckpt_path, exist_ok=True)

	# training
	for epoch in range(1, args.epochs+1):
		train_loss = train_one_epoch(
			args,
			epoch,
			online_encoder,
			online_encoder_without_ddp,
			target_encoder,
			target_ema_updater,
			barlow_twins_loss, 
			train_loader,
			optimizer,
			fp16_scaler,
			logger, 
			wandb_run,
		)
		if args.dataset == 'cifar10':
			if utils.is_main_process():
				test_acc_1, test_acc_5 = utils.eval_knn(online_encoder_without_ddp.backbone.encoder, memory_loader, test_loader, epoch, args.epochs, 10)
				if wandb_run is not None:
					wandb_run.log({'knn_test_acc_1': test_acc_1, 'knn_test_acc_5': test_acc_5})
		if epoch % args.epoch_save_f == 0 or epoch == args.epochs:
			save_dict = {
				'model': online_encoder.state_dict(),
				'optimizer': optimizer.state_dict(),
				'epoch': epoch + 1,
				'args': args,
				'barlow_twins_loss': barlow_twins_loss.state_dict(),
			}
			utils.save_on_master(
				save_dict,
				ckpt_path + f'/model_{epoch}.pth',
			)	
		if epoch % args.epoch_eval_f == 0 or epoch == args.epochs:
			if utils.is_main_process():
				if args.dataset == 'cifar10':
					pass
				else:
					eval_train_loader, eval_val_loader, eval_test_loader = get_fsd50k(args)
					scores = eval_linear(
						online_encoder_without_ddp.backbone.encoder,
						eval_train_loader,
						eval_val_loader,
						eval_test_loader, 
						args.use_fp16_eval,
					)
					score_all = scores.get('score_all')
					score_5 = scores.get('score_5')
					scores_teacher = eval_linear(
						target_encoder_without_ddp.backbone.encoder,
						eval_train_loader,
						eval_val_loader,
						eval_test_loader,
						args.use_fp16_eval,
					)
					score_all_teacher = scores_teacher.get('score_all')
					score_5_teacher = scores_teacher.get('score_5')
					if logger is not None:
						logger.info('epoch,{},step,{},linear_score,{},linear_score_5_mean,{},linear_score_5_std,{},linear_score_teacher,{},linear_score_teacher_5_mean,{},linear_score_teacher_5_std,{}'.format(
									epoch,len(train_loader)*epoch,score_all,score_5[0],score_5[1],
									score_all_teacher,score_5_teacher[0],score_5_teacher[1]))
					wandb_run.log({
						'FSD50K score (100%)': score_all,
						'FSD50K score (5pC) (mean)': score_5[0],
						'FSD50K score (teacher) (100%)': score_all_teacher,
						'FSD50K score (teacher) (5pC) (mean)': score_5_teacher[0],
					})
					
	

