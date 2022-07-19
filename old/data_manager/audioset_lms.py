"""
PyTorch dataset wrapper for the AudioSet dataset [Gemmeke et al., 2017]. 
Requires that the raw waveforms (.wav files) already converted to log mel spectrograms (.npy files). 
(i.e., All the data samples used here are expected to be `.npy` pre-converted spectrograms). 
Code adapted from https://github.com/nttcslab/msm-mae/blob/main/util/datasets.py.
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

from data_manager.transforms import make_transforms_pretrain_lms, make_transforms_eval_lms


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


class SpectrogramDataset(Dataset):
	"""
	Spectrogram audio dataset class.
	"""
	def __init__(
		self,
		cfg,
		n_views=2,
		base_dir="data/audioset_lms",
		crop_frames=96,
		transform=None,
		balanced_only=False,
		test=False,
	):
		super().__init__()

		# initializations 
		self.cfg = cfg 
		self.n_views = n_views 
		self.base_dir = base_dir 
		self.transform = transform 
		self.test = test 

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
				if self.cfg.data.audioset.twohundredk_only:
					self.combined_df = self.combined_df[:int(2e5)]
			
		# first column contains the audio fnames
		self.audio_fnames = np.asarray(self.combined_df.iloc[:, 0])
		# second column contains the labels (separated by # for multi-label)
		self.labels = np.asarray(self.combined_df.iloc[:, 1])
		# third column contains the identifier (balanced_train_segments or unbalanced_train_segments)
		self.ident = np.asarray(self.combined_df.iloc[:, 2])

		# load in class labels and create label -> index look-up dict 
		self.index_dict = make_index_dict(os.path.join(self.base_dir, "class_labels_indices.csv"))
		self.label_num = len(self.index_dict)

		# Norm stats
		self.norm_stats = self.cfg.data.preprocess.norm_stats


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

		# load .npy spectrograms 
		if ident == "balanced_train_segments":
			audio_fpath = os.path.join(os.path.join(*[self.base_dir, "balanced_train_segments", f"{audio_fname}.npy"]))
		elif ident == "unbalanced_train_segments":
			audio_fpath = os.path.join(os.path.join(*[self.base_dir, "unbalanced_train_segments", f"{audio_fname}.npy"]))
		elif ident == "eval_segments":
			audio_fpath = os.path.join(os.path.join(*[self.base_dir, "eval_segments", f"{audio_fname}.npy"]))

		lms = torch.tensor(np.load(audio_fpath)).unsqueeze(0)

		# Trim or pad
		l = lms.shape[-1]
		if l > self.crop_frames:
			start = np.random.randint(l - self.crop_frames)
			lms = lms[..., start:start + self.crop_frames]
		elif l < self.crop_frames:
			pad_param = []
			for i in range(len(lms.shape)):
				pad_param += [0, self.crop_frames - l] if i == 0 else [0, 0]
			lms = F.pad(lms, pad_param, mode='constant', value=0)
		lms = lms.to(torch.float)

		# Normalize
		if self.norm_stats is not None:
			lms = (lms - self.norm_stats[0]) / self.norm_stats[1]

		# Apply transforms
		if self.transform is not None:
			lms = [self.transform(lms) for _ in range(self.n_views)]

		if len(lms) == 1:
			return lms[0], label_indices
		else:
			return lms, label_indices


			
class SpectrogramLoader:

	def __init__(
		self,
		cfg,
		pretrain=True,
		finetune=False,
		balanced_only=False,
		test=False,
		num_workers=None,
	):
		self.cfg = cfg
		self.pretrain = pretrain
		self.finetune = finetune 
		self.balanced_only = balanced_only
		self.test = test
		self.num_workers = num_workers if num_workers is not None else self.cfg.data.dataloader.num_workers


	def get_loader(self, drop_last=True):
		# pretrain or downstream eval
		if self.pretrain:
			transform = make_transforms_pretrain_lms(self.cfg)
			dataset = SpectrogramDataset(
				self.cfg,
				n_views=2,
				transform=transform,
				balanced_only = self.balanced_only,
			)
		else:
			if self.finetune:
				transform = make_transforms_eval_lms(self.cfg)
			else:
				transform = None
			dataset = SpectrogramDataset(
				self.cfg,
				n_views=1,
				transform=transform,
				balanced_only=self.balanced_only,
				test=self.test,
			)

		if self.cfg.meta.distributed:
			sampler = torch.utils.data.distributed.DistributedSampler(dataset)
		else:
			sampler = None
			
		loader = DataLoader(
			dataset=dataset,
			batch_size=self.cfg.optimizer.batch_size_per_gpu,
			shuffle=(True if sampler is None else False),
			num_workers=self.num_workers,
			pin_memory=True,
			sampler=sampler,
			drop_last=drop_last,
		)

		return loader
