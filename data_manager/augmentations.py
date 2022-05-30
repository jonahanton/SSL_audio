"""

Audio augmentation modules.

Key:
	F: Number of frequency bins.
	T: Number of time frames.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT

import numpy as np
import random



class Mixup(nn.Module):
	"""
	Mixup augmentation, as defined in BYOL-A [Niizume et al., 2021]:
	"The Mixup block mised past randomly selected input audio in a small ratio."
	Code adapted from https://github.com/nttcslab/byol-a/blob/master/byol_a/augmentations.py.
	
	Args:
		ratio: Ratio 'alpha' that controls the degree of contrast between the mixed outputs.
		n_memory: Size of memory bank FIFO.
	"""

	def __init__(self, ratio=0.4, n_memory=2048):
		super().__init__()
		self.ratio = ratio
		self.n = n_memory
		self.memory_bank = []
		
		
	def forward(self, x):

		if isinstance(x, np.ndarray):
			x = torch.tensor(x)

		# mix random
		alpha = self.ratio * np.random.random()
		if self.memory_bank:
			# get z as a mixing background sound
			z = self.memory_bank[np.random.randint(len(self.memory_bank))]
			# mix them
			mixed = torch.log((1. - alpha)*x.exp() + alpha*z.exp() + torch.finfo(x.dtype).eps)
		else:
			mixed = x
		# update memory bank 
		self.memory_bank = (self.memory_bank + [x])[-self.n:]
		
		return mixed.to(torch.float)


class MixGaussianNoise(nn.Module):
	"""
	Mix with random (Gaussian) noise.
	"""

	def __init__(self, ratio=0.4):
		super().__init__()
		self.ratio = ratio
		
	
	def forward(self, x):

		if isinstance(x, np.ndarray):
			x = torch.tensor(x)
		
		lambd = self.ratio * np.random.random()
		# create random gaussian noise
		z = torch.normal(0, lambd, x.shape)
		
		# mix them
		mixed = torch.log((1. - lambd)*x.exp() + z.exp() + torch.finfo(x.dtype).eps)
		
		return mixed
		
	

if __name__ == "__main__":

	unit_length = int(0.95 * 16000)
	mixup = Mixup()
	gaussnoise = MixGaussianNoise()


	wav1, sr = torchaudio.load('data/audioset/samples/--1yd6dcNOQ.wav')
	wav2, sr = torchaudio.load('data/audioset/samples/MWTJo7DaBZQ.wav')
	wavs = [wav1, wav2]
	processed_wavs = []
	for wav in wavs:
		# if audio has 2 channels, convert to mono
		if wav.shape[0] == 2:
			wav = torch.mean(wav, dim=0).unsqueeze(0)
		wav = wav[0]  # (1, length) -> (length,)

		# zero padding to both ends
		length_adj = unit_length - len(wav)
		if length_adj > 0:
			half_adj = length_adj // 2
			wav = F.pad(wav, (half_adj, length_adj - half_adj))

		# random crop unit length wave
		length_adj = len(wav) - unit_length
		start = random.randint(0, length_adj) if length_adj > 0 else 0
		wav = wav[start:start + unit_length]

		processed_wavs.append(wav)
	
	wav1, wav2 = processed_wavs
	torchaudio.save('wav1.wav', wav1.unsqueeze(0), sample_rate=16000)
	torchaudio.save('wav2.wav', wav2.unsqueeze(0), sample_rate=16000)
	wav1_gauss = gaussnoise(wav1)
	wav2_gauss = gaussnoise(wav2)
	torchaudio.save('wav1_noisy.wav', wav1_gauss.unsqueeze(0), sample_rate=16000)
	torchaudio.save('wav2_noisy.wav', wav2_gauss.unsqueeze(0), sample_rate=16000)