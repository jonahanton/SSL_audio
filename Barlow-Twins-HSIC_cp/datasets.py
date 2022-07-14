import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as AT
from torch.utils.data import Dataset

import numpy as np
import random
import pandas as pd
import csv
import argparse
from tqdm import tqdm
import librosa



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
		# load in csv files
		if train:
			self.df = pd.read_csv("data/FSD50K/FSD50K.ground_truth/dev.csv", header=None)
		else:
			self.df = pd.read_csv("data/FSD50K/FSD50K.ground_truth/eval.csv", header=None)	
		self.files = np.asarray(self.df.iloc[:, 0])
		self.labels = np.asarray(self.df.iloc[:, 2])  # mids (separated by ,)
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
		# normalise lms with pre-computed dataset statistics
		if self.norm_stats is not None:
			lms = (lms - self.norm_stats[0]) / self.norm_stats[1]
		# transforms to lms
		if self.transform:
			lms = self.transform(lms)

		return lms, label_indices


def get_args_parser():
	
	parser = argparse.ArgumentParser(description='Calculate dataset normalization stats', add_help=False)
	parser.add_argument('--unit_sec', type=float, default=0.95)
	parser.add_argument('--sample_rate', type=int, default=16000)
	parser.add_argument('--n_fft', type=int, default=1024)
	parser.add_argument('--win_length', type=int, default=1024)
	parser.add_argument('--hop_length', type=int, default=160)
	parser.add_argument('--n_mels', type=int, default=64)
	parser.add_argument('--f_min', type=int, default=60)
	parser.add_argument('--f_max', type=int, default=7800)
	parser.add_argument('--n_norm_calc', type=int, default=10000)
	return parser


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


if __name__ == "__main__":

	parser = argparse.ArgumentParser('Norm-stats', parents=[get_args_parser()])
	args = parser.parse_args()
	# calculate norm stats
	calculate_norm_stats(args)