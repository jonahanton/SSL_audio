"""
Hyperparameter tuning using Optuna 
References:
	- https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
	- https://github.com/nttcslab/byol-a/blob/master/evaluate.py
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time

import optuna 
from optuna.trial import TrialState
import wandb

from utils.torch_mlp_clf import TorchMLPClassifier
from utils import transforms
import datasets
from model import BarlowTwins


parser = argparse.ArgumentParser(description='Train barlow twins')
parser.add_argument('--dataset', default='fsd50k', type=str, choices=['fsd50k'])
parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=5, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--optimizer', default='Adam', type=str, choices=['Adam'])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--model_type', default='resnet50', type=str, choices=['resnet50', 'resnet50_ReGP_NRF', 'audiontt'])
parser.add_argument('--lmbda', default=0.005, type=float, help='Lambda that controls the on- and off-diagonal terms')
parser.add_argument('--projector_out_dim', default=256, type=int)
parser.add_argument('--projector_n_hidden_layers', default=1, type=int)
parser.add_argument('--projector_hidden_dim', default=4096, type=int)
parser.add_argument('--unit_sec', type=float, default=0.95)
parser.add_argument('--crop_frames', type=int, default=96)
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--n_fft', type=int, default=1024)
parser.add_argument('--win_length', type=int, default=1024)
parser.add_argument('--hop_length', type=int, default=160)
parser.add_argument('--n_mels', type=int, default=64)
parser.add_argument('--f_min', type=int, default=60)
parser.add_argument('--f_max', type=int, default=7800)
parser.add_argument('--load_lms', action='store_true', default=True)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()


def objective(trial):
	
	# Generate the model
	model = define_model(trial).cuda()

	# Generate the optimizers 
	lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
	optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=lr)

	# Get FSD50K 
	train_loader, eval_train_loader, eval_val_loader, eval_test_loader = get_fsd50k(args)

	# Train the model 
	for epoch in range(1, args.epochs+1):
		model.train()
		print(f'Training [{epoch}/{args.epochs}]')
		loss = train_one_epoch(epoch, model, train_loader, optimizer)
		# Fit a linear classifier on frozen embeddings from encoder
		score = eval(model.encoder, eval_train_loader, eval_val_loader, eval_test_loader)
		trial.report(score, epoch)

		# Handle pruning based on the intermediate value
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()
	
	return score 


def define_model(trial):
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
	
	print('\nExtracting embeddings')
	start = time.time()
	X_train, y_train = get_embeddings(model, train_loader)
	X_val, y_val = get_embeddings(model, val_loader)
	X_test, y_test = get_embeddings(model, test_loader)
	print(f'Done\nTime elapsed = {time.time() - start:.2f}s')

	print('Fitting linear classifier')
	clf = TorchMLPClassifier(
		hidden_layer_sizes=(),
		max_iter=200,
		early_stopping=True,
		debug=True,
	)
	clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
	
	score = clf.score(X_test, y_test)
	return score


def train_one_epoch(epoch, model, data_loader, optimizer):
	model.train()
	total_loss, total_num, train_bar = 0, 0, tqdm(data_loader)
	for data_tuple in train_bar:
		(pos_1, pos_2), _ = data_tuple
		pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

		loss = model(pos_1, pos_2)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_num += args.batch_size
		total_loss += loss.item() * args.batch_size

		train_bar.set_description(f'BT Training Epoch: [{epoch}/{args.epochs}] '
								  f'Loss: {total_loss /total_num :.4f}')

	return total_loss / total_num


def get_fsd50k(args):
	norm_stats = [-4.950, 5.855]
	train_loader = DataLoader(
		datasets.FSD50K(args, split='train_val', transform=transforms.AudioPairTransform(train_transform = True), norm_stats=norm_stats), 
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


	study = optuna.create_study(direction='maximize')
	study.optimize(objective, n_trials=5)

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
