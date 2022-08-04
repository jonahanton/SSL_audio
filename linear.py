import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import datetime
import logging
from itertools import chain

from utils import hyperparameters
from utils.torch_mlp_clf import TorchMLPClassifier
import datasets
from model import BarlowTwins

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

MODELS = [
	'resnet50', 'resnet50_ReGP_NRF',
	'audiontt',
	'vit_base', 'vit_small', 'vit_tiny',
	'vitc_base', 'vitc_small', 'vitc_tiny',
]


def flatten_list(lists):
    return list(chain.from_iterable(lists))


@torch.no_grad()
def get_embeddings(model, data_loader):
	model.eval()
	embs, targets = [], []
	for data, target in data_loader:
		emb = model(data.cuda(non_blocking=True))
		if isinstance(emb, list):
			emb = emb[-1]
		emb = emb.detach().cpu().numpy()
		embs.extend(emb)
		targets.extend(target.numpy())

	return np.array(embs), np.array(targets)


def eval(model, train_loader, val_loader, test_loader):

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

	linear_score_all = clf.score(X_test, y_test)
	print(f'Done\tTime elapsed = {time.time() - start:.2f}s')

	# Extract random subsets containing 1, 2, 5 examples per class 
	print('Performing extreme low shot evaluation')
	print('Extracting (mutually exclusive) subsets')
	start = time.time()
	subset_1_1, subset_1_2, subset_1_5 = {}, {}, {}
	subset_2_1, subset_2_2, subset_2_5 = {}, {}, {}
	subset_3_1, subset_3_2, subset_3_5 = {}, {}, {}
	for idx, label in enumerate(y_train):
		classes = np.nonzero(label)[0]
		for c in classes:
			subset_1_1.setdefault(c, [])
			subset_2_1.setdefault(c, [])
			subset_3_1.setdefault(c, [])

			subset_1_2.setdefault(c, [])
			subset_2_2.setdefault(c, [])
			subset_3_2.setdefault(c, [])

			subset_1_5.setdefault(c, [])
			subset_2_5.setdefault(c, [])
			subset_3_5.setdefault(c, [])

			if len(subset_1_1[c]) < 1:
				subset_1_1[c].append(idx)
			elif len(subset_2_1[c]) < 1:
				subset_2_1[c].append(idx)
			elif len(subset_3_1[c]) < 1:
				subset_3_1[c].append(idx)

			if len(subset_1_2[c]) < 2:
				subset_1_2[c].append(idx)
			elif len(subset_2_2[c]) < 2:
				subset_2_2[c].append(idx)
			elif len(subset_3_2[c]) < 2:
				subset_3_2[c].append(idx)

			if len(subset_1_5[c]) < 5:
				subset_1_5[c].append(idx)
			elif len(subset_2_5[c]) < 5:
				subset_2_5[c].append(idx)
			elif len(subset_3_5[c]) < 5:
				subset_3_5[c].append(idx)

	subset_1_1 = np.unique(flatten_list([idxs for idxs in subset_1_1.values()]))
	subset_2_1 = np.unique(flatten_list([idxs for idxs in subset_2_1.values()]))
	subset_3_1 = np.unique(flatten_list([idxs for idxs in subset_3_1.values()]))

	subset_1_2 = np.unique(flatten_list([idxs for idxs in subset_1_2.values()]))
	subset_2_2 = np.unique(flatten_list([idxs for idxs in subset_2_2.values()]))
	subset_3_2 = np.unique(flatten_list([idxs for idxs in subset_3_2.values()]))

	subset_1_5 = np.unique(flatten_list([idxs for idxs in subset_1_5.values()]))
	subset_2_5 = np.unique(flatten_list([idxs for idxs in subset_2_5.values()]))
	subset_3_5 = np.unique(flatten_list([idxs for idxs in subset_3_5.values()]))

	print(f'Done\tTime elapsed = {time.time() - start:.2f}s')
	print('Fitting linear classifiers')
	start = time.time()

	clf.fit(X_train[subset_1_1], y_train[subset_1_1], X_val=X_val, y_val=y_val)
	linear_score_1_1 = clf.score(X_test, y_test)
	clf.fit(X_train[subset_2_1], y_train[subset_2_1], X_val=X_val, y_val=y_val)
	linear_score_2_1 = clf.score(X_test, y_test)
	clf.fit(X_train[subset_3_1], y_train[subset_3_1], X_val=X_val, y_val=y_val)
	linear_score_3_1 = clf.score(X_test, y_test)
	linear_score_1 = [linear_score_1_1, linear_score_2_1, linear_score_3_1]

	clf.fit(X_train[subset_1_2], y_train[subset_1_2], X_val=X_val, y_val=y_val)
	linear_score_1_2 = clf.score(X_test, y_test)
	clf.fit(X_train[subset_2_2], y_train[subset_2_2], X_val=X_val, y_val=y_val)
	linear_score_2_2 = clf.score(X_test, y_test)
	clf.fit(X_train[subset_3_2], y_train[subset_3_2], X_val=X_val, y_val=y_val)
	linear_score_3_2 = clf.score(X_test, y_test)
	linear_score_2 = [linear_score_1_2, linear_score_2_2, linear_score_3_2]

	clf.fit(X_train[subset_1_5], y_train[subset_1_5], X_val=X_val, y_val=y_val)
	linear_score_1_5 = clf.score(X_test, y_test)
	clf.fit(X_train[subset_2_5], y_train[subset_2_5], X_val=X_val, y_val=y_val)
	linear_score_2_5 = clf.score(X_test, y_test)
	clf.fit(X_train[subset_3_5], y_train[subset_3_5], X_val=X_val, y_val=y_val)
	linear_score_3_5 = clf.score(X_test, y_test)
	linear_score_5 = [linear_score_1_5, linear_score_2_5, linear_score_3_5]
	
	
	print(f'Done\tTime elapsed = {time.time() - start:.2f}s')

	results_dict = dict(
		linear_score_all = linear_score_all,
		linear_score_1 = (linear_score_1.mean(), linear_score_1.std()),
		linear_score_2 = (linear_score_2.mean(), linear_score_2.std()),
		linear_score_5 = (linear_score_5.mean(), linear_score_5.std()),
	)

	return results_dict


def get_data(args):
	if args.dataset == 'fsd50k':
		return get_fsd50k(args)


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


def log_print(msg):
	logger.info(msg)
	print(msg)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Linear eval', parents=hyperparameters.get_hyperparameters())
	parser.add_argument('--model_file_path', type=str, required=True)
	args = parser.parse_args()


	log_dir = f"logs/linear_eval/{args.dataset}/{args.name}/"
	os.makedirs(log_dir, exist_ok=True)
	timestamp = datetime.datetime.now().strftime('%H:%M_%h%d')
	log_path = os.path.join(log_dir, f"{timestamp}.log")
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)  # Setup the root logger
	logger.addHandler(logging.FileHandler(log_path, mode="w"))

	# Get data
	eval_train_loader, eval_val_loader, eval_test_loader = get_data(args)

	# Load model
	model = BarlowTwins(args).encoder
	sd = torch.load(args.model_file_path, map_location='cpu')
	sd_try = {k.replace("encoder.", "", 1): v for k, v in sd.items() if "encoder." in k}
	if len(sd_try) > 0:
		sd = sd_try
	model.load_state_dict(sd, strict=True)
	model = model.cuda()
	model.eval()

	# Linear evaluation 
	results = eval(model, eval_train_loader, eval_val_loader, eval_test_loader)
	log_print(f"Linear classification score (100% label fraction): {results['linear_score_all']}\n"
			  f"Extreme low-shot linear classification score scores\n"
			  f"\t1 example per class: {results['linear_score_1'][0]} +/- {results['linear_score_1'][1]}\n"
			  f"\t2 examples per class: {results['linear_score_2'][0]} +/- {results['linear_score_2'][1]}\n"
			  f"\t5 examples per class: {results['linear_score_5'][0]} +/- {results['linear_score_5'][1]}")