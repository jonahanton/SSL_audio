"""
Hyperparameter tuning using Optuna 
References:
	- https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
	- https://github.com/nttcslab/byol-a/blob/master/evaluate.py
"""

import argparse
import logging 
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
from easydict import EasyDict

import optuna 
from optuna.trial import TrialState
import wandb

from utils.torch_mlp_clf import TorchMLPClassifier
from utils import transforms
import datasets
from model import BarlowTwins


HYPERPARAMETERS = [
	'lr',
	'projector_n_hidden_layers',
	'projector_out_dim',
	'mixup_ratio',
	'virtual_crop_scale',
]


def get_std_config():
	cfg = EasyDict(
		epochs=5,
		batch_size=64,
		optimizer='Adam',
		lr=1e-4,
		model_type='resnet50',
		lmbda=0.005,
		projector_out_dim=256,
		projector_n_hidden_layers=1,
		projector_hidden_dim=4096,
		unit_sec=0.95,
		crop_frames=96,
		sample_rate=16000,
		n_fft=1024,
		win_length=1024,
		hop_length=160,
		n_mels=64,
		f_min=60,
		f_max=7800,
		load_lms=True,
		num_workers=4,
		mixup_ratio=0.2,
		virtual_crop_scale=(1, 1.5),
	)
	return cfg


def objective(trial):
	
	# Generate the model
	model = define_model(trial).cuda()

	# Generate the optimizers 
	if 'lr' in tune:
		args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
	optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)

	# Get FSD50K 
	train_loader, eval_train_loader, eval_val_loader, eval_test_loader = get_fsd50k(trial, args)

	# Train the model 
	for epoch in range(1, args.epochs+1):
		model.train()
		print(f'Training model with BT objective\n Epoch [{epoch}/{args.epochs}]')
		loss = train_one_epoch(epoch, model, train_loader, optimizer)
		print(f'Training epoch {epoch}/{args.epochs} finished')
		# Fit a linear classifier on frozen embeddings from encoder
		score = eval(model.encoder, eval_train_loader, eval_val_loader, eval_test_loader)
		trial.report(score, epoch)
		print(f'After epoch [{epoch}/{args.epochs}]\tTest score: {score:.4f}')

		# Handle pruning based on the intermediate value
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()
	
	return score 


def define_model(trial):
	if 'projector_n_hidden_layers' in tune:
		args.projector_n_hidden_layers = trial.suggest_int("projector_n_hidden_layers", 1, 2)
	if 'projector_out_dim' in tune:
		args.projector_out_dim = trial.suggest_categorical("projector_out_dim", [128, 256, 1024, 4096, 16384])
	return BarlowTwins(args)


@torch.no_grad()
def get_embeddings(model, data_loader):
	model.eval()
	embs, targets = [], []
	for data, target in data_loader:
		emb = model(data.cuda(non_blocking=True)).detach().cpu().numpy()
		embs.extend(emb)
		targets.extend(target.numpy())
	
	return np.array(embs), np.array(targets)


def eval(model, train_loader, val_loader, test_loader):
	
	print('Extracting embeddings from frozen backbone')
	start = time.time()
	X_train, y_train = get_embeddings(model, train_loader)
	X_val, y_val = get_embeddings(model, val_loader)
	X_test, y_test = get_embeddings(model, test_loader)
	print(f'Done\tTime elapsed = {time.time() - start:.2f}s')

	print('Fitting linear classifier')
	start = time.time()
	clf = TorchMLPClassifier(
		hidden_layer_sizes=(),
		max_iter=50,
		early_stopping=True,
		n_iter_no_change=10,
		debug=False,
	)
	clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
	
	score = clf.score(X_test, y_test)
	print(f'Done\tTime elapsed = {time.time() - start:.2f}s')

	return score


def train_one_epoch(epoch, model, data_loader, optimizer):
	model.train()
	total_loss, total_num = 0, 0
	start = time.time()
	for data_tuple in data_loader:
		(pos_1, pos_2), _ = data_tuple
		pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

		loss = model(pos_1, pos_2)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_num += args.batch_size
		total_loss += loss.item() * args.batch_size
		
	print(f'Done\tTime elapsed = {time.time() - start:.2f}s')

	return total_loss / total_num


def get_fsd50k(trial, args):

	if 'mixup_ratio' in tune:
		args.mixup_ratio = trial.suggest_float("mixup_ratio", 0, 1)
	if 'virtual_crop_scale' in tune:
		args.virtual_crop_scale = (trial.suggest_float("virtual_crop_scale_F", 1, 2), trial.suggest_float("virtual_crop_scale_T", 1, 2))

	norm_stats = [-4.950, 5.855]
	train_loader = DataLoader(
		datasets.FSD50K(args, split='train_val', 
						transform=transforms.AudioPairTransform(
							train_transform=True,
							mixup_ratio=args.mixup_ratio,
							virtual_crop_scale=args.virtual_crop_scale,
						), 
						norm_stats=norm_stats), 
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
	)
	eval_train_loader = DataLoader(
		datasets.FSD50K(args, split='train', transform=None, norm_stats=norm_stats), 
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	eval_val_loader = DataLoader(
		datasets.FSD50K(args, split='train', transform=None, norm_stats=norm_stats),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	eval_test_loader = DataLoader(
		datasets.FSD50K(args, split='test', transform=None, norm_stats=norm_stats),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	return train_loader, eval_train_loader, eval_val_loader, eval_test_loader


if __name__ == '__main__':

	args = get_std_config()
	parser = argparse.ArgumentParser(description='Hyperparameter tuning')
	parser.add_argument('--tune', nargs='+', type=str, default=['lr'], choices=HYPERPARAMETERS)
	parser.add_argument('--n_trials', type=int, default=5)
	cfg = parser.parse_args()
	tune, n_trials = cfg.tune, cfg.n_trials

	optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
	study = optuna.create_study(direction='maximize')
	study.optimize(objective, n_trials=n_trials)

	pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
	complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

	print("Study statistics: ")
	print("\tNumber of finished trials: ", len(study.trials))
	print("\tNumber of pruned trials: ", len(pruned_trials))
	print("\tNumber of complete trials: ", len(complete_trials))

	print("Best trial:")
	trial = study.best_trial
	print("\tValue: ", trial.value)
	print("\tParams: ")
	for key, value in trial.params.items():
		print("\t{}: {}".format(key, value))

