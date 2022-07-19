import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb 

import utils
import datasets
from model import BYOLAv2encoder, ResNet, ViT


if torch.cuda.is_available():
	torch.backends.cudnn.benchmark = True


def off_diagonal(x):
	# return a flattened view of the off-diagonal elements of a square matrix
	n, m = x.shape
	assert n == m
	return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def train_one_epoch(args, net, data_loader, train_optimizer, wandb_run):
	net.train()
	total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
	for data_tuple in train_bar:
		(pos_1, pos_2), _ = data_tuple
		pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

		feature_1, out_1 = net(pos_1)
		feature_2, out_2 = net(pos_2)
		
		# normalize the representations along the batch dimension
		out_1_norm = (out_1 - out_1.mean(dim=0)) / out_1.std(dim=0)
		out_2_norm = (out_2 - out_2.mean(dim=0)) / out_2.std(dim=0)
		
		# cross-correlation matrix
		c = torch.matmul(out_1_norm.T, out_2_norm) / args.batch_size
		# synchronise between gpus
		if args.distributed:
			torch.distributed.all_reduce(c)

		# loss
		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		off_diag = off_diagonal(c).pow_(2).sum()
		loss = on_diag + args.lmbda * off_diag

		train_optimizer.zero_grad()
		loss.backward()
		train_optimizer.step()

		total_num += args.batch_size
		total_loss += loss.item() * args.batch_size

		train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} lmbda:{:.4f} bsz:{} f_dim:{} dataset: {}'.format(\
								epoch, args.epochs, total_loss / total_num, args.lmbda, args.batch_size, args.feature_dim, args.dataset))

		if wandb_run is not None:
			wandb_run.log({'Loss': total_loss / total_num})

	return total_loss / total_num


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train barlow twins')
	parser.add_argument('--dataset', default='fsd50k', type=str, help='dataset',
						choices=['fsd50k', 'librispeech', 'fsd50k+librispeech', 'cifar10'])
	parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
	parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
	parser.add_argument('--epochs', default=20, type=int, help='Number of sweeps over the dataset to train')
	parser.add_argument('--epoch_save_f', default=20, type=int)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--wd', type=float, default=1e-6)
	# model type 
	parser.add_argument('--model_type', default='resnet', type=str, help='Encoder: resnet or byola or vit [tiny, small, base]')
	# for barlow twins
	parser.add_argument('--lmbda', default=0.005, type=float, help='Lambda that controls the on- and off-diagonal terms')
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
				project='barlow twins {}'.format(args.dataset),
				config=args,
				settings=wandb.Settings(start_method="fork"),
			)
	else:
		wandb_run = None
		
	# data prepare
	if args.dataset == 'cifar10':
		train_data = torchvision.datasets.CIFAR10(root='data', train=True, \
												  transform=utils.CifarPairTransform(train_transform=True), download=True)
	elif args.dataset == 'fsd50k':
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
	
	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
	else:
		train_sampler = None
		
	train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=(True if train_sampler is None else False),
							  num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

	# model setup and optimizer config
	if args.model_type == 'resnet':
		model = ResNet(args.feature_dim, args.dataset).cuda()
	elif args.model_type == 'vit_base':
		model = ViT(args.feature_dim, args.dataset, size='base', latent='cls').cuda()
	elif args.model_type == 'byola':
		model = BYOLAv2encoder(args.feature_dim, args.dataset, args.n_mels).cuda()
	save_name_pre = '{}_fdim{}_bs{}_{}'.format(args.model_type, args.feature_dim, args.batch_size, args.dataset)

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

	if 'vit' in args.model_type:
		optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd) 
	else:
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


	if not os.path.exists('results/{}'.format(args.dataset)):
		os.mkdir('results/{}'.format(args.dataset))


	for epoch in range(1, args.epochs+1):
		train_loss = train_one_epoch(args, model, train_loader, optimizer, wandb_run)
		if epoch % args.epoch_save_f == 0:
			utils.save_on_master(model.state_dict(), 'results/{}/{}_model_{}.pth'.format(args.dataset, save_name_pre, epoch))
