import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb 

import utils
from utils import LARS
import datasets
from model import BarlowTwins


if torch.cuda.is_available():
	torch.backends.cudnn.benchmark = True


def train_one_epoch(args, net, data_loader, train_optimizer, wandb_run):
	net.train()
	total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
	for data_tuple in train_bar:
		(pos_1, pos_2), _ = data_tuple
		pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

		loss = net(pos_1, pos_2)

		train_optimizer.zero_grad()
		loss.backward()
		train_optimizer.step()

		total_num += args.batch_size
		total_loss += loss.item() * args.batch_size

		train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} lmbda:{:.4f} bsz:{} dataset: {}'.format(\
								epoch, args.epochs, total_loss / total_num, args.lmbda, args.batch_size, args.dataset))

		if wandb_run is not None:
			wandb_run.log({'Loss': total_loss / total_num})

	return total_loss / total_num


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train barlow twins')
	parser.add_argument('--dataset', default='fsd50k', type=str, help='dataset',
						choices=['fsd50k', 'audioset', 'librispeech', 'fsd50k+librispeech'])
	parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
	parser.add_argument('--epochs', default=20, type=int, help='Number of sweeps over the dataset to train')
	parser.add_argument('--epoch_save_f', default=20, type=int)
	parser.add_argument('--optimizer', default='adam', type=str, choices = ['adam', 'adamw'])
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--wd', type=float, default=1e-6)
	# model type 
	parser.add_argument('--model_type', default='resnet50', type=str, choices=['resnet50', 'resnet50_ReGP_NRF', 'audiontt', 'vit'])
	# for barlow twins
	parser.add_argument('--lmbda', default=0.005, type=float, help='Lambda that controls the on- and off-diagonal terms')
	parser.add_argument('--projector_out_dim', default=8192, type=int)
	parser.add_argument('--projector_n_hidden_layers', default=1, type=int)
	parser.add_argument('--projector_hidden_dim', default=8192, type=int)
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
	# load pre-computed lms 
	parser.add_argument('--load_lms', action='store_true', default=True)
	# distributed training 
	parser.add_argument('--distributed', action='store_true', default=False)
	# data loader
	parser.add_argument('--num_workers', type=int, default=4)
	
	# args parse
	args = parser.parse_args()

	# distributed training 
	utils.init_distributed_mode(args)

	# wandb init
	if utils.is_main_process():
		wandb_run = wandb.init(
				project='barlow twins (orig) {}'.format(args.dataset),
				config=args,
				settings=wandb.Settings(start_method="fork"),
			)
	else:
		wandb_run = None
		
	# data prepare
	if args.dataset == 'fsd50k':
		# fsd50k [mean, std] (lms)
		norm_stats = [-4.950, 5.855]
		train_data = datasets.FSD50K(args, train=True, transform=utils.AudioPairTransform(train_transform = True), norm_stats=norm_stats)
	elif args.dataset == 'librispeech':
		# librispeech960 [mean, std] (lms)
		norm_stats = [-3.332, 4.205]
		train_data = datasets.LibriSpeech(args, train=True, transform=utils.AudioPairTransform(train_transform = True), norm_stats=norm_stats)
	elif args.dataset == 'fsd50k+librispeech':
		norm_stats_fsd50k = [-4.950, 5.855]
		norm_stats_librispeech = [-3.332, 4.205]
		train_data = torch.utils.data.dataset.ConcatDataset([
			datasets.FSD50K(args, train=True, transform=utils.AudioPairTransform(train_transform = True), norm_stats=norm_stats_fsd50k),
			datasets.LibriSpeech(args, train=True, transform=utils.AudioPairTransform(train_transform = True), norm_stats=norm_stats_librispeech),
		])
	elif args.dataset == 'audioset':
		norm_stats = [-0.8294, 4.6230]
		train_data = datasets.AudioSet(args, transform=utils.AudioPairTransform(train_transform = True), norm_stats=norm_stats)
	
	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
	else:
		train_sampler = None
		
	train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=(True if train_sampler is None else False),
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

	if args.optimizer == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) 
	elif args.optimzier == 'adamw':
		optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
	else:
		raise NotImplementedError(f'Optimizer {args.optimizer} not supported')

	# model checkpoint path
	ckpt_path = f'results/{args.dataset}/{args.model_type}'
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)


	for epoch in range(1, args.epochs+1):
		train_loss = train_one_epoch(args, model, train_loader, optimizer, wandb_run)
		if epoch % args.epoch_save_f == 0 or epoch == args.epochs:
			utils.save_on_master(model_without_ddp.encoder.state_dict(), ckpt_path + f'/model_{epoch}.pth')
