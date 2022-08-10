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
import math
import logging
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.trial import TrialState
import wandb

from utils.torch_mlp_clf import TorchMLPClassifier
from utils.loss import BarlowTwinsLoss
from utils import transforms, utils, hyperparameters
import datasets
from model import BarlowTwinsHead, ModelWrapper


HYPERPARAMETERS = [
	'lr', 'wd',
	'projector_n_hidden_layers',
	'projector_out_dim',
	'mixup_ratio',
	'virtual_crop_scale',
	'mask_beta',
]

CLASSES = dict(
	fsd50k=200,
	nsynth=88,
)


def objective(trial):

	# Generate the model
	model = define_model(trial).cuda()

	# prepare loss
	barlow_twins_loss = BarlowTwinsLoss(
		args,
		ncrops=args.local_crops_number+2,  # total number of crops = 2 global crops + local_crops_number
	).cuda()

	# Generate the optimizers
	if args.optimizer in ['Adam', 'AdamW', 'SGD']:
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

	elif args.optimizer == 'LARS':
		# separate lr for weights and biases using LARS optimizer
		if 'lr' in args.tune:
			lr_weights = trial.suggest_float("lr_weights", 1e-3, 1e0, log=True)
			lr_biases = trial.suggest_float("lr_biases", 1e-6, 1e-2, log=True)
		else:
			lr_weights = args.lr_weights
			lr_biases = args.lr_biases
		if 'wd' in args.tune:
			args.wd = trial.suggest_float("wd", 1e-8, 1e-4, log=True)
		param_weights = []
		param_biases = []
		for param in model.parameters():
			if param.ndim == 1:
				param_biases.append(param)
			else:
				param_weights.append(param)
		parameters = [
			{'params': param_weights, 'lr': lr_weights},
			{'params': param_biases, 'lr': lr_biases},
		]
		optimizer = utils.LARS(parameters, lr=0, weight_decay=args.wd,
			weight_decay_filter=True, lars_adaptation_filter=True)

	# Random masking ratio
	if 'mask_beta' in args.tune:
		assert args.mask
		assert args.mask_ratio_schedule
		args.mask_beta = trial.suggest_float("mask_beta", 0.05, 0.5)

	# Get data
	train_loader, eval_train_loader, eval_val_loader, eval_test_loader = get_data(trial)

	mask_ratio_scheduler = None
	if args.mask_ratio_schedule:
		mask_ratio_scheduler = utils.sine_scheduler_increase(
			final_value=args.mask_beta,
			epochs=args.train_epochs,
			niter_per_ep=len(train_loader),
			warmup_epochs=int(args.train_epochs / 5),
			warmup_value=0,
		)

	# mixed precision
	fp16_scaler = None
	if args.use_fp16:
		fp16_scaler = torch.cuda.amp.GradScaler()

	# Train the model
	print('Running training...')
	for epoch in range(1, args.train_epochs+1):
		model.train()
		loss = train_one_epoch(
			epoch,
			model,
			barlow_twins_loss,
			train_loader,
			optimizer,
			fp16_scaler,
			mask_ratio_scheduler,
		)
		# Report intermediate objective value
		score = evaluate(model.backbone.encoder, eval_train_loader, eval_val_loader, eval_test_loader)
		trial.report(score, epoch)

		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.TrialPruned()
	return score


def define_model(trial):
	if 'projector_n_hidden_layers' in args.tune:
		args.projector_n_hidden_layers = trial.suggest_categorical("projector_n_hidden_layers", [1, 2, 3])
	if 'projector_out_dim' in args.tune:
		args.projector_out_dim = trial.suggest_categorical("projector_out_dim", [64, 128, 256, 1024, 4096, 8192, 16384])
	
	model = ModelWrapper(args)
	model = utils.MultiCropWrapper(
		backbone=model,
		head=BarlowTwinsHead(
			args,
			in_dim=model.feature_dim,
		),
	)
	return model


def evaluate(model, train_loader, val_loader, test_loader):
	if args.eval == 'linear':
		return eval_linear(model, train_loader, val_loader, test_loader, args.use_fp16_eval)
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


def train_one_epoch(epoch, model, barlow_twins_loss, data_loader, optimizer, 
					fp16_scaler, mask_ratio_scheduler):
	model.train()
	total_loss, total_num, train_bar = 0, 0, tqdm(data_loader)

	total_data_time, total_forward_time, total_backward_time = 0, 0, 0
	tflag = time.time()
	for iteration, (images, _) in enumerate(train_bar):
		data_time = time.time() - tflag

		iteration += (epoch - 1) * len(data_loader)  # global training iteration

		# move images to gpu
		images = [im.cuda(non_blocking=True) for im in images]

		# mask ratio
		if args.mask:
			if mask_ratio_scheduler is not None:
				mask_ratio = mask_ratio_scheduler[iteration]
			else:
				mask_ratio = args.mask_ratio
		else:
			mask_ratio = 0

		tflag = time.time()
		# forward passes + compute barlow twins loss
		with torch.cuda.amp.autocast(enabled=(fp16_scaler is not None)):
			teacher_output = model(
				images[:1],  # only the 1 global crop passed through the teacher 
				mask_ratio=mask_ratio,
				ncrops=1,
			)
			# masked recon
			if args.masked_recon:
				teacher_output, recon_loss = teacher_output
			student_output = model(
				images[1:],  # 1 global crop + all local crops passed through the student
				ncrops=args.local_crops_number+1,
			)
			bt_loss = barlow_twins_loss(
				student_output,
				teacher_output,
				ngcrops_each=1,
			)
		forward_time = time.time() - tflag
		tflag = time.time()

		loss = bt_loss
		if args.masked_recon:
			loss += recon_loss


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
			intermediate_values.append([trial.number] + [q for p in params for q in p] + list(y))
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

	parser = argparse.ArgumentParser(description='All args', parents=hyperparameters.get_hyperparameters())
	parser.add_argument('--eval', type=str, default='linear', choices=['linear', 'knn'])
	parser.add_argument('--tune', nargs='+', type=str, default=['lr', 'wd'], choices=HYPERPARAMETERS)
	parser.add_argument('--n_trials', type=int, default=10)
	parser.add_argument('--train_epochs', type=int, default=20)
	args = parser.parse_args()
	hyperparameters.setup_hyperparameters(args)


	wandb_kwargs = dict(
		project=f'Hyperparameter sweep {args.model_type} [{args.dataset}]',
		config=args,
		name=f"{'_'.join(args.tune)} {args.name} - {args.n_trials} trials",
		settings=wandb.Settings(start_method="fork"),
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
