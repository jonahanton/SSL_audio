import torch
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
		if self.transform is not None:
			lms = self.transform(lms)

		return lms, label_indices


class LibriSpeech(Dataset):
	
	def __init__(self, cfg, train=True, transform=None, norm_stats=None):
		super().__init__()
		
		# initializations
		self.cfg = cfg
		self.train = train
		self.transform = transform
		self.norm_stats = norm_stats
		if self.cfg.load_lms:
			self.base_path= "data/LibriSpeech_lms/"
		else:
			self.base_path = "data/LibriSpeech/"

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
		# load in json file
		self.datapath = self.base_path + "librispeech_tr960_cut.json"
		with open(self.datapath, 'r') as fp:
			data_json = json.load(fp)
		self.data = data_json.get('data')
		

	def __len__(self):
		return len(self.data)
		
		
	def __getitem__(self, idx):
		datum = self.data[idx]
		fname = datum.get('wav')

		if self.cfg.load_lms:
			# load lms
			audio_path = self.base_path + fname[:-len(".flac")] + ".npy"
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
			# load raw audio
			audio_path = self.base_path + fname
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
		if self.transform is not None:
			lms = self.transform(lms)

		return lms, None


class NSynth(Dataset):
	
	def __init__(self, cfg, split='train', transform=None, norm_stats=None):
		super().__init__()
		
		# initializations
		self.cfg = cfg
		self.split = split
		self.transform = transform
		self.norm_stats = norm_stats
		if self.cfg.load_lms:
			self.base_path= "data/nsynth_lms/nsynth-{}/".format(split)
		else:
			self.base_path = "data/nsynth/nsynth-{}/".format(split) 

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
		# load in json file
		self.datapath = self.base_path + "examples.json"
		with open(self.datapath, 'r') as fp:
			data_json = json.load(fp)
		self.data = [(name, data.get("instrument_family")) for name, data in data_json.items()]
		
		
	def __len__(self):
		return len(self.data)
		
		
	def __getitem__(self, idx):
		fname, label = self.data[idx]

		if self.cfg.load_lms:
			# load lms
			audio_path = self.base_path + "audio/" + fname + ".npy"
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
			# load raw audio
			audio_path = self.base_path + "audio/" + fname + ".wav"
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
		if self.transform is not None:
			lms = self.transform(lms)

		return lms, label



def calculate_norm_stats(dataset, n_norm_calc=10000):

		# calculate norm stats (randomly sample n_norm_calc points from dataset)
		idxs = np.random.randint(0, len(dataset), size=n_norm_calc)
		lms_vectors = []
		for i in tqdm(idxs):
			lms_vectors.append(dataset[i][0])
		lms_vectors = torch.stack(lms_vectors)
		norm_stats = float(lms_vectors.mean()), float(lms_vectors.std() + torch.finfo().eps)

		print(f'Dataset contains {len(dataset)} files with normalizing stats\n'
			  f'mean: {norm_stats[0]}\t std: {norm_stats[1]}')
		norm_stats_dict = {'mean': norm_stats[0], 'std': norm_stats[1]}
		with open('norm_stats.json', mode='w') as jsonfile:
			json.dump(norm_stats_dict, jsonfile, indent=2)


if __name__ == "__main__":
	pass 