import torch
import torch.nn as nn
import augmentations 


class AudioPairTransform:
	def __init__(self, train_transform=True, pair_transform=True, 
				 mixup_ratio=0.2, virtual_crop_scale=(1.0, 1.5)):
		if train_transform is True:
			self.transform = nn.Sequential(
				augmentations.MixupBYOLA(ratio=mixup_ratio),
				augmentations.RandomResizeCrop(virtual_crop_scale=virtual_crop_scale),
				augmentations.RandomLinearFader(),
			)
		else:
			self.transform = nn.Identity()
		self.pair_transform = pair_transform 
	def __call__(self, x):
		if self.pair_transform is True:
			y1 = self.transform(x)
			y2 = self.transform(x)
			return y1, y2
		else:
			return self.transform(x)

