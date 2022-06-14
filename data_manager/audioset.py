"""
PyTorch dataset wrapper for the AudioSet dataset [Gemmeke et al., 2017]. 

References: 
	https://github.com/nttcslab/byol-a/blob/master/byol_a/dataset.py
	https://github.com/YuanGongND/ssast/tree/main
"""

import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as AT

from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import librosa
import random
import pandas as pd
import csv
import multiprocessing
import time
from pprint import pprint

from data_manager.transforms import make_transforms_pretrain, make_transforms_lineval


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


class AudioSet(Dataset):
	
	def __init__(
		self,
		cfg,
		n_views=2,
		base_dir="data/audioset",
		wav_transform=None,
		lms_transform=None,
		balanced_only=False,
		return_index=False,
		test=False,
	):
		
		super().__init__()
		
		# initializations
		self.cfg = cfg
		self.n_views = n_views
		self.base_dir = base_dir
		self.wav_transform = wav_transform 
		self.lms_transform = lms_transform
		self.balanced_only = balanced_only
		self.return_index = return_index
		self.test = test

		self.unit_length = int(cfg.data.preprocess.unit_sec * cfg.data.preprocess.sample_rate)
		self.to_melspecgram = AT.MelSpectrogram(
			sample_rate=cfg.data.preprocess.sample_rate,
			n_fft=cfg.data.preprocess.n_fft,
			win_length=cfg.data.preprocess.win_length,
			hop_length=cfg.data.preprocess.hop_length,
			n_mels=cfg.data.preprocess.n_mels,
			f_min=cfg.data.preprocess.f_min,
			f_max=cfg.data.preprocess.f_max,
			power=2,
		)
		
		# load in csv files
		self.unbalanced_df = pd.read_csv(
			os.path.join(self.base_dir, "unbalanced_train_segments-downloaded.csv"),
			header=None
		)
		self.balanced_df = pd.read_csv(
			os.path.join(self.base_dir, "balanced_train_segments-downloaded.csv"), 
			header=None
		)
		self.eval_df = pd.read_csv(
			os.path.join(self.base_dir, "eval_segments-downloaded.csv"),
			header=None
		)

		if self.test:
			self.combined_df = self.eval_df
		else:
			if self.balanced_only:
				self.combined_df = self.balanced_df
			else:
				self.combined_df = pd.concat([self.unbalanced_df, self.balanced_df], ignore_index=True)
			
		# first column contains the audio fnames
		self.audio_fnames = np.asarray(self.combined_df.iloc[:, 0])
		# second column contains the labels (separated by # for multi-label)
		self.labels = np.asarray(self.combined_df.iloc[:, 1])
		# third column contains the identifier (balanced_train_segments or unbalanced_train_segments)
		self.ident = np.asarray(self.combined_df.iloc[:, 2])

		# load in class labels and create label -> index look-up dict 
		self.index_dict = make_index_dict(os.path.join(self.base_dir, "class_labels_indices.csv"))
		self.label_num = len(self.index_dict)


	def __len__(self):
		return len(self.audio_fnames)
		
		
	def __getitem__(self, idx):
		audio_fname = self.audio_fnames[idx]
		labels = self.labels[idx]
		ident = self.ident[idx]

		# initialize the label
		label_indices = np.zeros(self.label_num)
		# add sample labels
		for label_str in labels.split('#'):
			label_indices[int(self.index_dict[label_str])] = 1.0
		label_indices = torch.FloatTensor(label_indices)

		# load .wav raw audio
		if ident == "balanced_train_segments":
			audio_fpath = os.path.join(os.path.join(*[self.base_dir, "balanced_train_segments", f"{audio_fname}.wav"]))
		elif ident == "unbalanced_train_segments":
			audio_fpath = os.path.join(os.path.join(*[self.base_dir, "unbalanced_train_segments", f"{audio_fname}.wav"]))
		elif ident == "eval_segments":
			audio_fpath = os.path.join(os.path.join(*[self.base_dir, "eval_segments", f"{audio_fname}.wav"]))

		wav, sr = torchaudio.load(audio_fpath)
		assert sr == self.cfg.data.preprocess.sample_rate, f"Convert .wav files to {self.cfg.data.preprocess.sample_rate} Hz. {audio_fname}.wav has {sr} Hz."
		
		# if audio has 2 channels, convert to mono
		if wav.shape[0] == 2:
			wav = torch.mean(wav, dim=0).unsqueeze(0)
		wav = wav[0]  # (1, length) -> (length,)
			
		# zero padding to both ends
		length_adj = self.unit_length - len(wav)
		if length_adj > 0:
			half_adj = length_adj // 2
			wav = F.pad(wav, (half_adj, length_adj - half_adj))
			
		# random crop unit length wave
		length_adj = len(wav) - self.unit_length
		start = random.randint(0, length_adj) if length_adj > 0 else 0
		wav = wav[start:start + self.unit_length]

		# transforms to raw waveform (must convert wav to np array)
		# note that transforms to raw waveform don't have cuda compatibility (done via audiomentations package, which uses librosa)		
		if self.wav_transform:
			wav = self.transform_wav(wav, n_views=self.n_views)
		else:
			wav = [wav]

		# to log mel spectogram -> (1, n_mels, time)
		lms = self.convert_to_melspecgram(wav)
		
		# transforms to lms
		if self.lms_transform:
			lms = self.transform_lms(lms)

		if len(lms) == 1:
			if self.return_index:
				return lms[0], label_indices, idx
			else:
				return lms[0], label_indices
		else:
			if self.return_index:
				return lms, label_indices, idx
			else:
				return lms, label_indices

	
	def transform_wav(self, wav, n_views):
		out = []
		for n in range(n_views):
			w_tf = self.wav_transform(samples=wav.numpy(), sample_rate=self.cfg.data.preprocess.sample_rate)
			w_tf = torch.tensor(w_tf)
			out.append(w_tf)
		return out

	
	def convert_to_melspecgram(self, wav):
		out = []
		for w in wav:
			lms = (self.to_melspecgram(w) + torch.finfo().eps).log().unsqueeze(0)
			out.append(lms)
		return out

	
	def transform_lms(self, lms):
		out = []
		for l in lms:
			l_tf = self.lms_transform(l)
			out.append(l)
		return out
		
			
class AudioSetLoader:

	def __init__(
		self,
		cfg,
		pretrain=True,
		balanced_only=False,
		return_index=False,
		test=False,
	):
		self.cfg = cfg
		self.pretrain = pretrain
		self.balanced_only = balanced_only
		self.return_index = return_index
		self.test = test

	def get_loader(self, drop_last=True):
		# pretrain or downstream eval
		if self.pretrain:
			wav_transform, lms_transform = make_transforms_pretrain(self.cfg)
			dataset = AudioSet(
				self.cfg,
				n_views=2,
				wav_transform=wav_transform,
				lms_transform=lms_transform,
				balanced_only = self.balanced_only,
				return_index=self.return_index,
			)
		else:
			if self.test:
				wav_transform, lms_transform = None, None
			else:
				wav_transform, lms_transform = make_transforms_lineval(self.cfg)
			dataset = AudioSet(
				self.cfg,
				n_views=1,
				wav_transform=wav_transform,
				lms_transform=lms_transform,
				balanced_only=self.balanced_only,
				return_index=self.return_index,
				test=self.test,
			)

		# if self.cfg.meta.distributed:
		sampler = torch.utils.data.distributed.DistributedSampler(dataset)
			
		loader = DataLoader(
			dataset=dataset,
			batch_size=self.cfg.optimizer.batch_size_per_gpu,
			shuffle=False,
			num_workers=self.cfg.data.dataloader.num_workers,
			pin_memory=True,
			sampler=sampler,
			drop_last=drop_last,
		)

		# else:
		# 	loader = DataLoader(
		# 		dataset=dataset,
		# 		batch_size=self.cfg.optimizer.batch_size_per_gpu,
		# 		shuffle=True,
		# 		num_workers=self.cfg.data.dataloader.num_workers,
		# 		pin_memory=True,
		# 		drop_last=False,
		# 	)

		return loader



if __name__ == "__main__":
	
	base_dir="data/audioset"
	index_dict = make_index_dict(os.path.join(base_dir, "class_labels_indices.csv"))
	label_num = len(index_dict)
	labels = "/m/01v_m0#/m/0hdsk"
	# initialize the label
	label_indices = np.zeros(label_num)
	# add sample labels
	for label_str in labels.split('#'):
		label_indices[int(index_dict[label_str])] = 1.0
	label_indices = torch.FloatTensor(label_indices)
	pprint(label_indices)