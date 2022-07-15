import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as AT
from torch.utils.data import Dataset

import nnAudio.features

import numpy as np
import random
import pandas as pd
import csv
import argparse
from tqdm import tqdm
import librosa
import json


def make_index_dict(label_csv):
	index_lookup = {}
	with open(label_csv, 'r') as f:
		csv_reader = csv.DictReader(f)
		for row in csv_reader:
			index_lookup[row['mids']] = row['index']
	return index_lookup


class FSD50K(Dataset):
	
	def __init__(self, cfg, train=True, transform=None, norm_stats=None):
		super().__init__()
		
		# initializations
		self.cfg = cfg
		self.train = train
		self.transform = transform
		self.norm_stats = norm_stats

		self.unit_length = int(cfg.unit_sec * cfg.sample_rate)
		self.to_melspecgram = AT.MelSpectrogram(
			sample_rate=cfg.sample_rate,
			n_fft=cfg.n_fft,
			win_length=cfg.win_length,
			hop_length=cfg.hop_length,
			n_mels=cfg.n_mels,
			f_min=cfg.f_min,
			f_max=cfg.f_max,
			power=2,
		)
		# self.to_melspecgram = nnAudio.features.mel.MelSpectrogram(
		# 	sr=cfg.sample_rate,
		# 	n_fft=cfg.n_fft,
		# 	win_length=cfg.win_length,
		# 	hop_length=cfg.hop_length,
		# 	n_mels=cfg.n_mels,
		# 	fmin=cfg.f_min,
		# 	fmax=cfg.f_max,
		# 	center=True,
		# 	power=2,
		# 	verbose=False,
		# )
		# load in csv files
		if train:
			self.df = pd.read_csv("data/FSD50K/FSD50K.ground_truth/dev.csv", header=None)
		else:
			self.df = pd.read_csv("data/FSD50K/FSD50K.ground_truth/eval.csv", header=None)	
		self.files = np.asarray(self.df.iloc[:, 0], dtype=str)
		self.labels = np.asarray(self.df.iloc[:, 2], dtype=str)  # mids (separated by ,)
		self.index_dict = make_index_dict("data/FSD50K/FSD50K.ground_truth/vocabulary.csv")
		self.label_num = len(self.index_dict)


	def __len__(self):
		return len(self.files)
		
		
	def __getitem__(self, idx):
		fname = self.files[idx]
		labels = self.labels[idx]
		# initialize the label
		label_indices = np.zeros(self.label_num)
		# add sample labels
		for label_str in labels.split(','):
			label_indices[int(self.index_dict[label_str])] = 1.0
		label_indices = torch.FloatTensor(label_indices)
		if self.cfg.load_lms:
			# load lms
			if self.train:
				audio_path = "data/FSD50K_lms/FSD50K.dev_audio/" + fname + ".npy"
			else:
				audio_path = "data/FSD50K_lms/FSD50K.eval_audio/" + fname + ".npy"
			lms = torch.tensor(np.load(audio_path)).unsqueeze(0)
			# Trim or pad
			l = lms.shape[-1]
			if l > self.cfg.crop_frames:
				start = np.random.randint(l - self.cfg.crop_frames)
				lms = lms[..., start:start + self.cfg.crop_frames]
			elif l < self.cfg.crop_frames:
				pad_param = []
				for i in range(len(lms.shape)):
					pad_param += [0, self.cfg.crop_frames - l] if i == 0 else [0, 0]
				lms = F.pad(lms, pad_param, mode='constant', value=0)
			lms = lms.to(torch.float)
		else:
<<<<<<< HEAD
			audio_path = "data/FSD50K/FSD50K.eval_audio/" + fname + ".wav"
		wav, org_sr = librosa.load(audio_path, sr=self.cfg.sample_rate)
		wav = torch.tensor(wav)  # (length,)
		# zero padding to both ends
		length_adj = self.unit_length - len(wav)
		if length_adj > 0:
			half_adj = length_adj // 2
			wav = F.pad(wav, (half_adj, length_adj - half_adj))
		# random crop unit length wave
		length_adj = len(wav) - self.unit_length
		start = random.randint(0, length_adj) if length_adj > 0 else 0
		wav = wav[start:start + self.unit_length]
		# to log mel spectogram -> (1, n_mels, time)
		lms = (self.to_melspecgram(wav) + torch.finfo().eps).log()
		lms = lms.unsqueeze(0)  # if using torchaudio.transforms.MelSpectrogram
=======
			# load raw audio
			if self.train:
				audio_path = "data/FSD50K/FSD50K.dev_audio/" + fname + ".wav"
			else:
				audio_path = "data/FSD50K/FSD50K.eval_audio/" + fname + ".wav"
			wav, org_sr = librosa.load(audio_path, sr=self.cfg.sample_rate)
			wav = torch.tensor(wav)  # (length,)
			# zero padding to both ends
			length_adj = self.unit_length - len(wav)
			if length_adj > 0:
				half_adj = length_adj // 2
				wav = F.pad(wav, (half_adj, length_adj - half_adj))
			# random crop unit length wave
			length_adj = len(wav) - self.unit_length
			start = random.randint(0, length_adj) if length_adj > 0 else 0
			wav = wav[start:start + self.unit_length]
			# to log mel spectogram -> (1, n_mels, time)
			lms = (self.to_melspecgram(wav) + torch.finfo().eps).log()
			lms = lms.unsqueeze(0)
>>>>>>> 86c639bc81c341d7c05edefd9de57a87d7923b95
		# normalise lms with pre-computed dataset statistics
		if self.norm_stats is not None:
			lms = (lms - self.norm_stats[0]) / self.norm_stats[1]
		# transforms to lms
		if self.transform:
			lms = self.transform(lms)

		return lms, label_indices


def calculate_norm_stats(args):

		# load dataset
		dataset = FSD50K(args)

		# calculate norm stats (randomly sample n_norm_calc points from dataset)
		idxs = np.random.randint(0, len(dataset), size=args.n_norm_calc)
		lms_vectors = []
		for i in tqdm(idxs):
			lms_vectors.append(dataset[i][0])
		lms_vectors = torch.stack(lms_vectors)
		norm_stats = lms_vectors.mean(), lms_vectors.std() + torch.finfo().eps

		print(f'Dataset contains {len(dataset)} files with normalizing stats\n'
			f'mean: {norm_stats[0]}\t std: {norm_stats[1]}')
		norm_stats_dict = {'mean': norm_stats[0], 'std': norm_stats[1]}
		with open('norm_stats.json', mode='w') as jsonfile:
			json.dump(norm_stats_dict, jsonfile, indent=2)


if __name__ == "__main__":

<<<<<<< HEAD
=======
	
	def get_args_parser():
		
		parser = argparse.ArgumentParser(description='Train barlow twins')
		parser.add_argument('--dataset', default='fsd50k', type=str, help='Dataset: fsd50k or cifar10 or tiny_imagenet or stl10')
		parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
		parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
		parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
		parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
		parser.add_argument('--epochs', default=20, type=int, help='Number of sweeps over the dataset to train')
		# for barlow twins
		parser.add_argument('--lmbda', default=0.005, type=float, help='Lambda that controls the on- and off-diagonal terms')
		parser.add_argument('--corr_neg_one', dest='corr_neg_one', action='store_true')
		parser.add_argument('--corr_zero', dest='corr_neg_one', action='store_false')
		parser.set_defaults(corr_neg_one=False)
		parser.add_argument('--unit_sec', type=float, default=0.95)
		parser.add_argument('--sample_rate', type=int, default=16000)
		parser.add_argument('--n_fft', type=int, default=1024)
		parser.add_argument('--win_length', type=int, default=1024)
		parser.add_argument('--hop_length', type=int, default=160)
		parser.add_argument('--n_mels', type=int, default=64)
		parser.add_argument('--f_min', type=int, default=60)
		parser.add_argument('--f_max', type=int, default=7800)
		parser.add_argument('--n_norm_calc', type=int, default=10000)
		# load pre-computed lms 
		parser.add_argument('--load_lms', action='store_true', default=False)
		return parser

	
>>>>>>> 86c639bc81c341d7c05edefd9de57a87d7923b95
	def off_diagonal(x):
		# return a flattened view of the off-diagonal elements of a square matrix
		n, m = x.shape
		assert n == m
		return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

	parser = argparse.ArgumentParser(description='Train barlow twins')
	parser.add_argument('--dataset', default='fsd50k', type=str, help='Dataset: fsd50k or cifar10 or tiny_imagenet or stl10')
	parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
	parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
	parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
	parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
	parser.add_argument('--epochs', default=20, type=int, help='Number of sweeps over the dataset to train')
	# for barlow twins
	parser.add_argument('--lmbda', default=0.005, type=float, help='Lambda that controls the on- and off-diagonal terms')
	parser.add_argument('--corr_neg_one', dest='corr_neg_one', action='store_true')
	parser.add_argument('--corr_zero', dest='corr_neg_one', action='store_false')
	parser.set_defaults(corr_neg_one=False)
	parser.add_argument('--unit_sec', type=float, default=0.95)
	parser.add_argument('--sample_rate', type=int, default=16000)
	parser.add_argument('--n_fft', type=int, default=1024)
	parser.add_argument('--win_length', type=int, default=1024)
	parser.add_argument('--hop_length', type=int, default=160)
	parser.add_argument('--n_mels', type=int, default=64)
	parser.add_argument('--f_min', type=int, default=60)
	parser.add_argument('--f_max', type=int, default=7800)
	parser.add_argument('--n_norm_calc', type=int, default=10000)

	args = parser.parse_args()

	dataset = args.dataset
	feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
	batch_size, epochs = args.batch_size, args.epochs
	lmbda = args.lmbda
	corr_neg_one = args.corr_neg_one

	from torch.utils.data import DataLoader
	import torch.optim as optim
	from model import Model
	import utils
	from tqdm import tqdm

	model = Model(feature_dim, dataset).cuda()
	optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

	norm_stats = [-4.950, 5.855]
	train_data = FSD50K(args, train=True, transform=utils.FSD50KPairTransform(train_transform = True), norm_stats=norm_stats)
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

	import time 
	data_time, forward_time, loss_time, backward_time = [], [], [], []
	tflag = time.time()
	for it, data_tuple in tqdm(enumerate(train_loader)):

		if it >= 10:
			break
		data_time.append(time.time() - tflag) 

		(pos_1, pos_2), _ = data_tuple
		pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
		
		tflag = time.time()
		feature_1, out_1 = model(pos_1)
		feature_2, out_2 = model(pos_2)
		forward_time.append(time.time() - tflag)

		tflag = time.time()

		# Barlow Twins
		
		# normalize the representations along the batch dimension
		out_1_norm = (out_1 - out_1.mean(dim=0)) / out_1.std(dim=0)
		out_2_norm = (out_2 - out_2.mean(dim=0)) / out_2.std(dim=0)
		
		# cross-correlation matrix
		c = torch.matmul(out_1_norm.T, out_2_norm) / batch_size

		# loss
		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		if corr_neg_one is False:
			# the loss described in the original Barlow Twin's paper
			# encouraging off_diag to be zero
			off_diag = off_diagonal(c).pow_(2).sum()
		else:
			# inspired by HSIC
			# encouraging off_diag to be negative ones
			off_diag = off_diagonal(c).add_(1).pow_(2).sum()
		loss = on_diag + lmbda * off_diag

		loss_time.append(time.time() - tflag)

		tflag = time.time()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		backward_time.append(time.time() - tflag)

		tflag = time.time()

	print(f'Data time: mean {np.mean(data_time)} std {np.std(data_time)}\n'
		  f'Forward time: mean {np.mean(forward_time)} std {np.std(forward_time)}\n'
		  f'Loss time: mean {np.mean(loss_time)} std {np.std(loss_time)}\n'
		  f'Backward time: mean {np.mean(backward_time)} std {np.std(backward_time)}\n')