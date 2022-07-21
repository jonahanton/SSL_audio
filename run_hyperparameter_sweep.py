"""
References:
	- https://github.com/nttcslab/byol-a/blob/master/evaluate.py
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb 
import optuna 
from sklearn.preprocessing import StandardScaler
import numpy as np

from utils.torch_mlp_clf import TorchMLPClassifier
from utils import transforms
import datasets
from model import BarlowTwins



@torch.no_grad()
def get_embeddings(net, data_loader):
	net.eval()
	embs, labels = [], []
	for X, label in tqdm(data_loader):
		emb = net(X.cuda(non_blocking=True)).detach().cpu().numpy()
		embs.extend(emb)
		labels.extend(label.numpy())
	
	return np.array(embs), np.array(labels)


def eval(net, train_loader, val_loader, test_loader):
	
	X_train, y_train = get_embeddings(net, train_loader)
	X_val, y_val = get_embeddings(net, val_loader)
	X_test, y_test = get_embeddings(net, test_loader)

	clf = TorchMLPClassifier(
		hidden_layer_sizes=(),
		max_iter=200,
		early_stopping=True,
		debug=True,
	)
	clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
	
	score = clf.score(X_test, y_test)
	return score


def train_one_epoch(args, epoch, net, data_loader, train_optimizer):
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

	return total_loss / total_num


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train barlow twins')
	parser.add_argument('--dataset', default='fsd50k', type=str, choices=['fsd50k'])
	parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
	parser.add_argument('--epochs', default=20, type=int, help='Number of sweeps over the dataset to train')
	parser.add_argument('--epoch_save_f', default=20, type=int)
	parser.add_argument('--optimizer', default='adam', type=str, choices=['adam'])
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--wd', type=float, default=1e-6)
	# model type 
	parser.add_argument('--model_type', default='resnet50', type=str, choices=['resnet50', 'resnet50_ReGP_NRF', 'audiontt'])
	# for barlow twins
	parser.add_argument('--lmbda', default=0.005, type=float, help='Lambda that controls the on- and off-diagonal terms')
	parser.add_argument('--projector_out_dim', default=256, type=int)
	parser.add_argument('--projector_n_hidden_layers', default=1, type=int)
	parser.add_argument('--projector_hidden_dim', default=4096, type=int)
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
	# data loader
	parser.add_argument('--num_workers', type=int, default=4)
	
	# args parse
	args = parser.parse_args()


	# data prepare
	if args.dataset == 'fsd50k':
		# fsd50k [mean, std] (lms)
		norm_stats = [-4.950, 5.855]
		train_data = datasets.FSD50K(args, split='train_val', transform=transforms.AudioPairTransform(train_transform = True), norm_stats=norm_stats)
		eval_train_data = datasets.FSD50K(args, split='train', transform=None, norm_stats=norm_stats)
		eval_val_data = datasets.FSD50K(args, split='val', transform=None, norm_stats=norm_stats)
		eval_test_data = datasets.FSD50K(args, split='test', transform=None, norm_stats=norm_stats)
	else:
		raise NotImplementedError(f'Dataset {args.dataset} not supported for hyperparameter tuning')

	train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
							num_workers=args.num_workers, pin_memory=True, drop_last=True)
	eval_train_loader = DataLoader(eval_train_data, batch_size=args.batch_size, shuffle=True,
							num_workers=args.num_workers, pin_memory=True, drop_last=False)
	eval_val_loader = DataLoader(eval_val_data, batch_size=args.batch_size, shuffle=True,
							num_workers=args.num_workers, pin_memory=True, drop_last=False)
	eval_test_loader = DataLoader(eval_test_data, batch_size=args.batch_size, shuffle=True,
							num_workers=args.num_workers, pin_memory=True, drop_last=False)

	# model setup 
	model = BarlowTwins(args).cuda()

	if args.optimizer == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) 
	else:
		raise NotImplementedError(f'Optimizer {args.optimizer} not supported')

	for epoch in range(1, args.epochs+1):
		train_loss = train_one_epoch(args, epoch, model, train_loader, optimizer)
		test_mAP = eval(model.encoder, eval_train_loader, eval_val_loader, eval_test_loader)
		print(f'Test mAP: {test_mAP}')


