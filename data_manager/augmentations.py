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


class RandomResizeCrop(nn.Module):
	"""
	Random Resize Crop block, as defined in BYOL-A [Niizume et al., 2021].
	Code taken from https://github.com/nttcslab/byol-a/blob/master/byol_a/augmentations.py.
	Args:
		virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
		freq_scale: Random frequency range `(min, max)`.
		time_scale: Random time frame range `(min, max)`.
	"""

	def __init__(self, virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)):
		super().__init__()
		self.virtual_crop_scale = virtual_crop_scale
		self.freq_scale = freq_scale
		self.time_scale = time_scale
		self.interpolation = 'bicubic'
		assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

	@staticmethod
	def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
		canvas_h, canvas_w = virtual_crop_size
		src_h, src_w = in_size
		h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
		w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
		i = random.randint(0, canvas_h - h) if canvas_h > h else 0
		j = random.randint(0, canvas_w - w) if canvas_w > w else 0
		return i, j, h, w

	def forward(self, lms):
		# make virtual_crop_arear empty space (virtual crop area) and copy the input log mel spectrogram to th the center
		virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
		virtual_crop_area = (torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))
								.to(torch.float).to(lms.device))
		_, lh, lw = virtual_crop_area.shape
		c, h, w = lms.shape
		x, y = (lw - w) // 2, (lh - h) // 2
		virtual_crop_area[:, y:y+h, x:x+w] = lms
		# get random area
		i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
		crop = virtual_crop_area[:, i:i+h, j:j+w]
		# print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
		lms = F.interpolate(crop.unsqueeze(0), size=lms.shape[-2:],
							mode=self.interpolation, align_corners=True).squeeze(0)
		return lms.to(torch.float)


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
		
