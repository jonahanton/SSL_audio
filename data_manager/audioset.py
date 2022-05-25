"""
PyTorch dataset wrapper for the AudioSet dataset [Gemmeke et al., 2017]. 

Adapted from https://github.com/nttcslab/byol-a/blob/master/byol_a/dataset.py.

"""

import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as AT

from torch.utils.data import Dataset

import os
import numpy as np
import librosa
import random
import pandas as pd


class AudioSet(Dataset):
	
	def __init__(self, cfg, base_dir, wav_transform=None, lms_transform=None):
		
		super().__init__()
		
		# initializations
		self.cfg = cfg
		self.base_dir = base_dir
		self.wav_transform = wav_transform 
		self.lms_transform = lms_transform
		self.unit_length = int(cfg.unit_sec * cfg.sample_rate)
		self.to_melspecgram  AT.MelSpectrogram(
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
		self.unbalanced_df = pd.read_csv(
			os.path.join(self.base_dir, "unbalanced_train_segments-downloaded.csv"),
			header=None
		)
		self.balanced_df = pd.read_csv(
			os.path.join(self.base_dir, "balanced_train_segments-downloaded.csv"), 
			header=None
		)
		self.combined_df = pd.concat([self.unbalanced_df, self.balanced_df], ignore_index=True)
		
		# first column contains the audio fnames
		self.audio_fnames = np.asarray(self.combined_df.iloc[:, 0])
		# second column contains the labels (separated by # for multi-label)
		self.labels = np.asarray(self.combined_df.iloc[:, 1])
		# third column contains the identifier (balanced_train_segments or unbalanced_train_segments)
		self.ident = np.asarray(self.combined_df.iloc[;, 2])
	
		def __len__(self):
			return len(self.audio_fnames)
		
		
		def __getitem__(self, idx):
			# load .wav audio
			audio_fname = self.audio_fnames[idx]
			if self.ident[idx] == "balanced_train_segments":
				audio_fpath = os.path.join(os.path.join(*[self.base_dir, "balanced_train_segments", f"{audio_fname}.wav"]))
			else:
				audio_fpath = os.path.join(os.path.join(*[self.base_dir, "unbalanced_train_segments", f"{audio_fname}.wav"]))
				
			wav, sr = torchaudio.load(audio_fpath)
			assert sr = self.cfg.sample_rate, f"Convert .wav files to {self.cfg.sample_rate} Hz. {audio_fname}.wav has {sr} Hz."
			
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
				wav = self.wav_transform(wav.numpy())
				wav = torch.tensor(wav)
				
			# to log mel spectogram -> (1, n_mels, time)
			lms = (self.to_melspecgram(wav) + torch.finfo().eps).log().unsqueeze(0)
			
			# transform to lms
			if self.lms_transform:
				lms = self.lms_transform(lms)
				
			return lms
			
			
