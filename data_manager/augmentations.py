"""
Audio augmentation modules.
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
		
