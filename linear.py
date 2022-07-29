import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import datetime
import logging

from utils.torch_mlp_clf import TorchMLPClassifier
import datasets
from model import BarlowTwins


MODELS = [
	'resnet50', 'resnet50_ReGP_NRF',
	'audiontt',
	'vit_base', 'vit_small', 'vit_tiny',
	'vitc_base', 'vitc_small', 'vitc_tiny',
]


def get_std_parser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--lmbda', type=float, default=0.005)
	parser.add_argument('--alpha', type=float, default=1)
	parser.add_argument('--projector_out_dim', default=8192, type=int)
	parser.add_argument('--projector_n_hidden_layers', default=2, type=int)
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
	parser.add_argument('--HSIC', action='store_true', default=False)
	parser.add_argument('--squeeze_excitation', action='store_true', default=False)
	parser.add_argument('--load_lms', action='store_true', default=True)
	return parser


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
		max_iter=200,
		early_stopping=True,
		n_iter_no_change=10,
		debug=True,
	)
	clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

	score = clf.score(X_test, y_test)
	print(f'Done\tTime elapsed = {time.time() - start:.2f}s')

	return score


def get_data():
	if args.dataset == 'fsd50k':
		return get_fsd50k(args)


def get_fsd50k():
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


def log_print(msg):
	logger.info(msg)
	print(msg)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Hyperparameter tuning', parents=[get_std_parser()])
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--dataset', type=str, default='fsd50k', choices=['fsd50k'])
	parser.add_argument('--model_type', type=str, default='audiontt', choices=MODELS)
	parser.add_argument('--model_file_path', type=str, required=True)
	parser.add_argument('--name', type=str, default='')
	args = parser.parse_args()


	log_dir = f"logs/linear_eval/{args.dataset}/{args.name}/"
	os.makedirs(log_dir, exist_ok=True)
	timestamp = datetime.datetime.now().strftime('%H:%M_%h%d')
	log_path = os.path.join(log_dir, f"{'_'.join(args.tune)}_{timestamp}.log")
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)  # Setup the root logger
	logger.addHandler(logging.FileHandler(log_path, mode="w"))

	# Get data
	eval_train_loader, eval_val_loader, eval_test_loader = get_data()

	# Load model
	model = BarlowTwins(args).encoder
	model.load_state_dict(torch.load(args.model_file_path, map_location='cpu'), strict=False)
	model = model.cuda()
	model.eval()

	# Linear evaluation 
	eval_linear(eval_train_loader, eval_val_loader, eval_test_loader)
