"""
Data augmentations on log mel spectrogram, taken from BYOL-A(v2)
Copy-paste from https://github.com/nttcslab/byol-a/blob/master/v2/byol_a2/augmentations.py
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random 


class RandomResizeCrop(nn.Module):
	"""Random Resize Crop block.
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

	def __repr__(self):
		format_string = self.__class__.__name__ + f'(virtual_crop_size={self.virtual_crop_scale}'
		format_string += ', time_scale={0}'.format(tuple(round(s, 4) for s in self.time_scale))
		format_string += ', freq_scale={0})'.format(tuple(round(r, 4) for r in self.freq_scale))
		return format_string


class RandomLinearFader(nn.Module):
	def __init__(self, gain=1.0):
		super().__init__()
		self.gain = gain

	def forward(self, lms):
		head, tail = self.gain * ((2.0 * np.random.rand(2)) - 1.0) # gain * U(-1., 1) for two ends
		T = lms.shape[2]
		slope = torch.linspace(head, tail, T, dtype=lms.dtype).reshape(1, 1, T).to(lms.device)
		y = lms + slope # add liniear slope to log-scale input
		return y

	def __repr__(self):
		format_string = self.__class__.__name__ + f'(gain={self.gain})'
		return format_string


def log_mixup_exp(xa, xb, alpha):
	xa = xa.exp()
	xb = xb.exp()
	x = alpha * xa + (1. - alpha) * xb
	return torch.log(x + torch.finfo(x.dtype).eps)


class MixupBYOLA(nn.Module):
	"""Mixup for BYOL-A.
	Args:
		ratio: Alpha in the paper.
		n_memory: Size of memory bank FIFO.
		log_mixup_exp: Use log-mixup-exp to mix if this is True, or mix without notion of log-scale.
	"""

	def __init__(self, ratio=0.2, n_memory=2048, log_mixup_exp=True):
		super().__init__()
		self.ratio = ratio
		self.n = n_memory
		self.log_mixup_exp = log_mixup_exp
		self.memory_bank = []

	def forward(self, x):
		# mix random
		alpha = self.ratio * np.random.random()
		if self.memory_bank:
			# get z as a mixing background sound
			z = self.memory_bank[np.random.randint(len(self.memory_bank))]
			# mix them
			mixed = log_mixup_exp(x, z, 1. - alpha) if self.log_mixup_exp \
					else alpha * z + (1. - alpha) * x
		else:
			mixed = x
		# update memory bank
		self.memory_bank = (self.memory_bank + [x])[-self.n:]

		return mixed.to(torch.float)

	def __repr__(self):
		format_string = self.__class__.__name__ + f'(ratio={self.ratio},n={self.n}'
		format_string += f',log_mixup_exp={self.log_mixup_exp})'
		return format_string
