import torch
import torch.nn as nn
import augmentations 


class AudioPairTransform(nn.Module):
	
	def __init__(self, args, train_transform=True, pair_transform=True, 
				 mixup_ratio=0.2, gauss_noise_ratio=0.3, virtual_crop_scale=(1.0, 1.5)):
		super().__init__()
		if train_transform is True:
			transforms = []
			if args.mixup:
				transforms.append(augmentations.MixupBYOLA(ratio=mixup_ratio))
			elif args.Gnoise:
				transforms.append(augmentations.MixGaussianNoise(ratio=gauss_noise_ratio))
			if args.RRC:
				transforms.append(augmentations.RandomResizeCrop(virtual_crop_scale=virtual_crop_scale))
			if args.RLF:
				transforms.append(augmentations.RandomLinearFader())
			self.transform = nn.Sequential(*transforms)
		else:
			self.transform = nn.Identity()
		self.pair_transform = pair_transform 

	def forward(self, x):
		if self.pair_transform is True:
			y1 = self.transform(x)
			y2 = self.transform(x)
			return y1, y2
		else:
			return self.transform(x)

