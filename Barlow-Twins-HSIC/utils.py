from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import augmentations 
import builtins
import time
import datetime
import torch.distributed as dist
import os
import torch


# for audio
class AudioPairTransform:
	def __init__(self, train_transform = True, pair_transform=True):
		if train_transform is True:
			self.transform = nn.Sequential(
				augmentations.MixupBYOLA(),
				augmentations.RandomResizeCrop(),
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

# for cifar10 (32x32)
class CifarPairTransform:
	def __init__(self, train_transform = True, pair_transform = True):
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

# for tiny imagenet (64x64)
class TinyImageNetPairTransform:
	def __init__(self, train_transform = True, pair_transform = True):
		if train_transform is True:
			self.transform = transforms.Compose([
					transforms.RandomApply(
						[transforms.ColorJitter(brightness=0.4, contrast=0.4, 
												saturation=0.4, hue=0.1)], 
						p=0.8
					),
					transforms.RandomGrayscale(p=0.1),
					transforms.RandomResizedCrop(
						64,
						scale=(0.2, 1.0),
						ratio=(0.75, (4 / 3)),
						interpolation=Image.BICUBIC,
					),
					transforms.RandomHorizontalFlip(p=0.5),
					transforms.ToTensor(),
					transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
				])
		else:
			self.transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
				])
		self.pair_transform = pair_transform
	def __call__(self, x):
		if self.pair_transform is True:
			y1 = self.transform(x)
			y2 = self.transform(x)
			return y1, y2
		else:
			return self.transform(x)

# for stl10 (96x96)
class StlPairTransform:
	def __init__(self, train_transform = True, pair_transform = True):
		if train_transform is True:
			self.transform = transforms.Compose([
					transforms.RandomApply(
						[transforms.ColorJitter(brightness=0.4, contrast=0.4, 
												saturation=0.4, hue=0.1)], 
						p=0.8
					),
					transforms.RandomGrayscale(p=0.1),
					transforms.RandomResizedCrop(
						64,
						scale=(0.2, 1.0),
						ratio=(0.75, (4 / 3)),
						interpolation=Image.BICUBIC,
					),
					transforms.RandomHorizontalFlip(p=0.5),
					transforms.ToTensor(),
					transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
				])
		else:
			self.transform = transforms.Compose([
					transforms.Resize(70, interpolation=Image.BICUBIC),
					transforms.CenterCrop(64),
					transforms.ToTensor(),
					transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
				])
		self.pair_transform = pair_transform
	def __call__(self, x):
		if self.pair_transform is True:
			y1 = self.transform(x)
			y2 = self.transform(x)
			return y1, y2
		else:
			return self.transform(x)



# utils for distributed training
def init_distributed_mode(cfg):

	if cfg.distributed:

		cfg.rank = int(os.environ['RANK'])
		cfg.gpu = int(os.environ['LOCAL_RANK'])
		cfg.world_size = int(os.environ['WORLD_SIZE'])
		env_dict = {
			key: os.environ[key]
			for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
		}
		print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

		dist.init_process_group(backend='nccl')
		print(
			f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
			+ f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
		)
		dist.barrier()
	else:
		cfg.rank, cfg.gpu, cfg.world_size = 0, 0, 1
		
	torch.cuda.set_device(cfg.gpu)
	setup_for_distributed(cfg.rank == 0)


def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def setup_for_distributed(is_master):
	"""
	This function disables printing when not in master process
	"""
	builtin_print = builtins.print

	def print(*args, **kwargs):
		force = kwargs.pop('force', False)
		force = force or (get_world_size() > 8)
		if is_master or force:
			now = datetime.datetime.now().time()
			builtin_print('[{}] '.format(now), end='')  # print with time stamp
			builtin_print(*args, **kwargs)

	builtins.print = print


def save_on_master(*args, **kwargs):
	if is_main_process():
		torch.save(*args, **kwargs)


def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()


def is_main_process():
	return get_rank() == 0


def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()


