"""
Hyperparameter tuning using Optuna 
References:
	- https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
	- https://github.com/nttcslab/byol-a/blob/master/evaluate.py
"""

import argparse
import logging 
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import datetime
import csv
import logging
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


import optuna 
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.trial import TrialState
import wandb

from utils.torch_mlp_clf import TorchMLPClassifier
from utils import transforms, utils
import datasets
from model import BarlowTwins


HYPERPARAMETERS = [
	'lr', 'wd',
	'projector_n_hidden_layers',
	'projector_out_dim',
	'mixup_ratio',
	'virtual_crop_scale',
]

MODELS = [
	'resnet50', 'resnet50_ReGP_NRF',
	'audiontt',
	'vit_base', 'vit_small', 'vit_tiny',
	'vitc_base', 'vitc_small', 'vitc_tiny',
]

CLASSES = dict(
	fsd50k=200,
	nsynth=88,
)


def get_std_parser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--lmbda', type=str, default=0.005)
	parser.add_argument('--projector_out_dim', default=8192, type=int)
	parser.add_argument('--projector_n_hidden_layers', default=1, type=int)
	parser.add_argument('--projector_hidden_dim', default=8192, type=int)
	parser.add_argument('--unit_sec', type=float, default=0.95)
	parser.add_argument('--crop_frames', type=int, default=96)
	parser.add_argument('--sample_rate', type=int, default=16000)
	parser.add_argument('--n_fft', type=int, default=1024)
	parser.add_argument('--win_length', type=int, default=1024)
	parser.add_argument('--hop_length', type=int, default=160)
	parser.add_argument('--n_mels', type=int, default=64)
	parser.add_argument('--f_min', type=int, default=60)
	parser.add_argument('--f_max', type=int, default=7800)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--mixup_ratio', type=float, default=0.2)
	parser.add_argument('--virtual_crop_scale', nargs='+', type=float, default=[1, 1.5])
	parser.add_argument('--HSIC', action='store_true', default=False)
	return parser


def objective(trial):
	
	# Generate the model
	model = define_model(trial).cuda()

	# Generate the optimizers 
	if 'lr' in args.tune:
		args.lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
	if 'wd' in args.tune:
		args.wd = trial.suggest_float("wd", 1e-3, 1e0, log=True)

	if args.optimizer == 'Adam':
		optimizer = optim.Adam(utils.get_param_groups(model), lr=args.lr, weight_decay=args.wd)
	elif args.optimizer == 'AdamW':
		optimizer = optim.AdamW(utils.get_param_groups(model), lr=args.lr, weight_decay=args.wd)
	elif args.optimizer == 'SGD':
		optimizer = optim.SGD(utils.get_param_groups(model), lr=args.lr, weight_decay=args.wd)

	# Get data
	train_loader, eval_train_loader, eval_val_loader, eval_test_loader = get_data(trial)

	# Train the model 
	print('Running training...')
	for epoch in range(1, args.train_epochs+1):
		model.train()
		loss = train_one_epoch(epoch, model, train_loader, optimizer)
		# Report intermediate objective value
		score = evaluate(model.encoder, eval_train_loader, eval_val_loader, eval_test_loader)
		trial.report(score, epoch)

		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.TrialPruned()
	return score 


def define_model(trial):
	if 'projector_n_hidden_layers' in args.tune:
		args.projector_n_hidden_layers = trial.suggest_int("projector_n_hidden_layers", 1, 2)
	if 'projector_out_dim' in args.tune:	
		args.projector_out_dim = trial.suggest_categorical("projector_out_dim", [64, 128, 256, 1024, 4096])
	return BarlowTwins(args)


def evaluate(model, train_loader, val_loader, test_loader):
	if args.eval == 'linear':
		return eval_linear(model, train_loader, val_loader, test_loader)
	elif args.eval == 'knn':
		return eval_knn(model, train_loader, test_loader)


@torch.no_grad()
def eval_knn(model, memory_data_loader, test_data_loader, k=200, temperature=0.5):
	"""
	kNN accuracy - Copy-paste from https://github.com/yaohungt/Barlow-Twins-HSIC/blob/main/main.py
	"""
	model.eval()
	c = CLASSES[args.dataset]
	total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
	# generate feature bank and target bank
	for data_tuple in tqdm(memory_data_loader, desc='Feature extracting'):
		data, target = data_tuple
		target_bank.append(target)
		feature = model(data.cuda(non_blocking=True))
		feature_bank.append(feature)
	# [D, N]
	feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
	# [N]
	feature_labels = torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
	# loop test data to predict the label by weighted knn search
	test_bar = tqdm(test_data_loader)
	for data_tuple in test_bar:
		data, target = data_tuple
		data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
		feature = model(data)

		total_num += data.size(0)
		# compute cos similarity between each feature vector and feature bank ---> [B, N]
		sim_matrix = torch.mm(feature, feature_bank)
		# [B, K]
		sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
		# [B, K]
		sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
		sim_weight = (sim_weight / temperature).exp()

		# counts for each class
		one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
		# [B*K, C]
		one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
		# weighted score ---> [B, C]
		pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

		pred_labels = pred_scores.argsort(dim=-1, descending=True)
		total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
		total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
		test_bar.set_description('Acc@1:{:.2f}% Acc@5:{:.2f}%'
									.format(total_top1 / total_num * 100, total_top5 / total_num * 100))

	return total_top1 / total_num


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
		hidden_layer_sizes=(),
		max_iter=100,
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
	total_loss, total_num, train_bar = 0, 0, tqdm(data_loader)
	
	total_data_time, total_forward_time, total_backward_time = 0, 0, 0
	tflag = time.time()
	for data_tuple in train_bar:
		data_time = time.time() - tflag
		tflag = time.time()

		(pos_1, pos_2), _ = data_tuple
		pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

		loss = model(pos_1, pos_2)
		forward_time = time.time() - tflag 
		tflag = time.time()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		backward_time = time.time() - tflag 

		total_num += args.batch_size
		total_loss += loss.item() * args.batch_size

		total_data_time += data_time
		total_forward_time += forward_time 
		total_backward_time += backward_time

		train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Data time {:.2f}({:.2f}) Forward time {:.2f}({:.2f}) Backward time {:.2f}({:.2f}))'.format(
								  epoch, args.train_epochs, total_loss / total_num, 
								  data_time, total_data_time,
								  forward_time, total_forward_time,
								  backward_time, total_backward_time))
		
		tflag = time.time()
		
	return total_loss / total_num


def get_data(trial):
	if args.dataset == 'fsd50k':
		return get_fsd50k(trial)
	elif args.dataset == 'nsynth':
		return get_nsynth_50h(trial)


def get_nsynth_50h(trial):

	if 'mixup_ratio' in args.tune:
		args.mixup_ratio = trial.suggest_categorical("mixup_ratio", [0, 0.2, 0.4, 0.6, 0.8])
	if 'virtual_crop_scale' in args.tune:
		args.virtual_crop_scale = args.virtual_crop_scale = (
			trial.suggest_categorical("virtual_crop_scale_F", [1, 1.2, 1.4, 1.6, 1.8]), 
			trial.suggest_categorical("virtual_crop_scale_T", [1, 1.2, 1.4, 1.6, 1.8]),
		)

	norm_stats = [-8.82, 7.03]
	train_loader = DataLoader(
		datasets.NSynth_HEAR(args, split='train', 
						transform=transforms.AudioPairTransform(
							args,
							train_transform=True,
							mixup_ratio=args.mixup_ratio,
							virtual_crop_scale=args.virtual_crop_scale,
						), 
						norm_stats=norm_stats), 
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
	)
	eval_train_loader = DataLoader(
		datasets.NSynth_HEAR(args, split='train', transform=None, norm_stats=norm_stats), 
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	eval_val_loader = DataLoader(
		datasets.NSynth_HEAR(args, split='valid', transform=None, norm_stats=norm_stats), 
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	eval_test_loader = DataLoader(
		datasets.NSynth_HEAR(args, split='test', transform=None, norm_stats=norm_stats),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	return train_loader, eval_train_loader, eval_val_loader, eval_test_loader


def get_fsd50k(trial):

	if 'mixup_ratio' in args.tune:
		args.mixup_ratio = trial.suggest_categorical("mixup_ratio", [0, 0.2, 0.4, 0.6, 0.8])
	if 'virtual_crop_scale' in args.tune:
		args.virtual_crop_scale = (
			trial.suggest_categorical("virtual_crop_scale_F", [1, 1.2, 1.4, 1.6, 1.8]), 
			trial.suggest_categorical("virtual_crop_scale_T", [1, 1.2, 1.4, 1.6, 1.8]),
		)

	norm_stats = [-4.950, 5.855]
	train_loader = DataLoader(
		datasets.FSD50K(args, split='train_val', 
						transform=transforms.AudioPairTransform(
							args,
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
		datasets.FSD50K(args, split='val', transform=None, norm_stats=norm_stats),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	eval_test_loader = DataLoader(
		datasets.FSD50K(args, split='test', transform=None, norm_stats=norm_stats),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False,
	)
	return train_loader, eval_train_loader, eval_val_loader, eval_test_loader


def log_print(msg):
	logger.info(msg)
	print(msg)


def plot_and_save_intermediate_values(study, save_path):
	target_state = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING]
	trials = [trial for trial in study.trials if trial.state in target_state]
	intermediate_values = []
	for trial in trials:
		if trial.intermediate_values:
			sorted_intermediate_values = sorted(trial.intermediate_values.items()) 
			x=tuple((x for x, _ in sorted_intermediate_values))
			y=tuple((y for _, y in sorted_intermediate_values))
			params = [(k,v) for k,v in trial.params.items()]
			label_str = ','.join([f'{p[0]}={p[1]}' for p in params])
			intermediate_values.append([trial.number] + [q for p in params for q in p] + list(x))
			plt.plot(x, y, marker='o', label=label_str)
	plt.xlabel('Epoch')
	plt.ylabel('Score')
	plt.title('Intermediate scores')
	plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
	plt.tight_layout()
	plt.show()
	plt.savefig(os.path.join(save_path, 'intermediate_values.png'), bbox_inches = 'tight')
	with open(os.path.join(save_path, 'intermediate_values.csv'), 'w') as f:
		writer = csv.writer(f)
		writer.writerows(intermediate_values)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Hyperparameter tuning', parents=[get_std_parser()])
	parser.add_argument('--dataset', type=str, default='fsd50k', choices=['fsd50k', 'nsynth'])
	parser.add_argument('--eval', type=str, default='linear', choices=['linear', 'knn'])
	parser.add_argument('--tune', nargs='+', type=str, default=['lr', 'wd'], choices=HYPERPARAMETERS)
	parser.add_argument('--n_trials', type=int, default=10)
	parser.add_argument('--train_epochs', type=int, default=20)
	parser.add_argument('--model_type', type=str, default='audiontt', choices=MODELS)
	parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW', 'SGD'])
	parser.add_argument('--load_lms', action='store_true', default=True)
	parser.add_argument('--load_wav', dest='load_lms', action='store_false')
	parser.add_argument('--mixup', action='store_true', default=True)
	parser.add_argument('--no_mixup', action='store_false', dest='mixup')
	parser.add_argument('--RRC', action='store_true', default=True)
	parser.add_argument('--no_RRC', action='store_false', dest='RRC')
	parser.add_argument('--RLF', action='store_true', default=True)
	parser.add_argument('--no_RLF', action='store_false', dest='RLF')
	parser.add_argument('--Gnoise', action='store_true', default=False)
	parser.add_argument('--name', type=str, default='')
	args = parser.parse_args()

	if args.optimizer == 'Adam' or 'SGD':
		args.wd = 0
	elif args.optimizer == 'AdamW':
		args.wd = 1e-2

	wandb_kwargs = dict(
		project=f'Hyperparameter sweep {args.model_type} [{args.dataset}]',
		config=args,
		name=f"{'_'.join(args.tune)} {args.name} - {args.n_trials} trials",
	)
	wandbc = WeightsAndBiasesCallback(
		metric_name='score',
		wandb_kwargs=wandb_kwargs,
	)

	log_dir = f"logs/hparams/{args.dataset}/{args.model_type}{args.name}/" 
	os.makedirs(log_dir, exist_ok=True)
	timestamp = datetime.datetime.now().strftime('%H:%M_%h%d')
	log_path = os.path.join(log_dir, f"{'_'.join(args.tune)}_{timestamp}.log")
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)  # Setup the root logger
	logger.addHandler(logging.FileHandler(log_path, mode="w"))
	optuna.logging.set_verbosity(optuna.logging.INFO)
	optuna.logging.enable_propagation()  # Propagate logs to the root logger

	study = optuna.create_study(
		direction='maximize',
		sampler=optuna.samplers.TPESampler(),
		pruner=optuna.pruners.HyperbandPruner(),
	)
	study.optimize(objective, n_trials=args.n_trials, callbacks=[wandbc])

	pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
	complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

	log_print("Study statistics: ")
	log_print(f"\tNumber of finished trials: {len(study.trials)}")
	log_print(f"\tNumber of pruned trials: {len(pruned_trials)}")
	log_print(f"\tNumber of complete trials: {len(complete_trials)}")

	log_print("Best trial:")
	trial = study.best_trial
	log_print(f"\tValue: {trial.value}")
	wandb.run.summary["best top1"] = trial.value
	log_print("\tParams: ")
	for key, value in trial.params.items():
		log_print("\t{}: {}".format(key, value))
		wandb.run.summary[key] = value

	optimization_history = optuna.visualization.plot_optimization_history(study)
	intermediate_values = optuna.visualization.plot_intermediate_values(study)
	param_importances = optuna.visualization.plot_param_importances(study)
	param_relationships = optuna.visualization.plot_parallel_coordinate(study)
	param_slices = optuna.visualization.plot_slice(study)
	wandb.log(
			{
				"optimization_history": optimization_history,
				"intermediate_values": intermediate_values,
				"param_importances": param_importances,
				"param_relationships": param_relationships,
				"param_slices": param_slices,
			}
		)
	wandb.finish()

	plot_and_save_intermediate_values(study, save_path=log_dir)
