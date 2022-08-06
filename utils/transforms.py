import torch
import torch.nn as nn
from torchvision import transforms
import augmentations 


class AudioPairTransform(nn.Module):

	def __init__(self, args, train_transform=True, multi_transform=True, 
				 mixup_ratio=0.2, gauss_noise_ratio=0.2,
				 global_crop_scale=(0.6, 1.5), local_crop_scale=(0.05, 0.6)):
		super().__init__()
		self.multi_transform = multi_transform
		self.local_crops_number = args.local_crops_number
		
		if train_transform is True:
			global_transforms = []
			if args.mixup:
				global_transforms.append(augmentations.MixupBYOLA(ratio=mixup_ratio))
			if args.Gnoise:
				global_transforms.append(augmentations.MixGaussianNoise(ratio=gauss_noise_ratio))
			if args.RRC:
				out_size = (args.n_mels, args.crop_frames)
				global_transforms.append(
					augmentations.RandomResizeCrop(
						out_size,
						virtual_crop_scale=tuple(args.virtual_crop_scale),
						freq_scale=global_crop_scale,
						time_scale=global_crop_scale,
					)
				)
			if args.RLF:
				global_transforms.append(augmentations.RandomLinearFader())
			self.global_transform = nn.Sequential(*global_transforms)
		else:
			self.global_transform = nn.Identity()

		local_transforms = []
		local_transforms.append(
			augmentations.RandomResizeCrop(
				tuple(args.local_crops_size),
				virtual_crop_scale=(1, 1),
				freq_scale=local_crop_scale,
				time_scale=local_crop_scale,
			)
		)
		self.local_transform = nn.Sequential(*local_transforms)

	def forward(self, x):
		if self.multi_transform:
			crops = []
			crops.append(self.global_transform(x))
			crops.append(self.global_transform(x))
			for _ in range(self.local_crops_number):
				crops.append(self.local_transform(x))
			return crops
		else:
			return self.global_transform(x)



class CifarPairTransform(nn.Module):
	def __init__(self, train_transform = True, pair_transform = True):
		super().__init__()
		if train_transform is True:
			self.transform = transforms.Compose([
				transforms.RandomResizedCrop(32),
				transforms.RandomHorizontalFlip(p=0.5),
				transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
				transforms.RandomGrayscale(p=0.2),
				transforms.ToTensor(),
				transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
		else:
			self.transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
		self.pair_transform = pair_transform
	def __call__(self, x):
		if self.pair_transform is True:
			y1 = self.transform(x)
			y2 = self.transform(x)
			return y1, y2
		else:
			return self.transform(x)