import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

from vision_transformer import mae
import utils 


def off_diagonal(x):
	# return a flattened view of the off-diagonal elements of a square matrix
	n, m = x.shape
	assert n == m
	return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
	
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._setup_model()
	

	def _setup_model(self):
		
		if self.cfg.model_type == 'resnet50':
			self.encoder = AudioResNet50(self.cfg)
		elif self.cfg.model_type == 'audiontt':
			self.encoder = AudioNTT2022(self.cfg)
		elif self.cfg.model_type == 'vit':
			self.encoder = ViT(self.cfg)
		else:
			raise NotImplementedError(f'Model type {self.cfg.model_type} is not supported')
		feature_dim = self.encoder.embed_dim
		
		sizes = [feature_dim, 4*feature_dim, 4*feature_dim, 4*feature_dim]
		layers = []
		for i in range(len(sizes) - 2):
			layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
			layers.append(nn.BatchNorm1d(sizes[i + 1]))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
		self.projector = nn.Sequential(*layers)

		self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
	

	def forward(self, y1, y2):
		
		feature1 = self.encoder(y1)
		feature2 = self.encoder(y2)

		z1 = self.projector(feature1)
		z2 = self.projector(feature2)
		
		# empirical cross-correlation matrix
		c = self.bn(z1).T @ self.bn(z2)
		
		# sum the cross-correlation matrix between all gpus
		c.div_(z1.shape[0])
		if utils.is_dist_avail_and_initialized():
			torch.distributed.all_reduce(c)
		
		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		off_diag = off_diagonal(c).pow_(2).sum()
		loss = on_diag + self.cfg.lambd * off_diag
		return loss


class AudioResNet50(nn.Module):
	def __init__(self):
		super().__init__()

		convs = []
		for name, module in resnet50().named_children():
			if name == 'conv1':
				module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
			if not isinstance(module, nn.Linear):
				convs.append(module)
		self.features = nn.Sequential(*convs)
		self.embed_dim = 2048

	def forward(self, x):
		out = self.features(x)
		out = torch.flatten(x, start_dim=1)
		return out


class ViT(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.encoder = mae.mae_vit_base_patch16x16()
		self.embed_dim = self.encoder.embed_dim

	def forward(self, x):
		x = self.encoder(x)
		if self.cfg.latent == 'cls':
			x = x[:, 0]
		else:
			x = torch.mean(x[:, 1:], dim=1)
		feature = x.contiguous()
		return feature


class AudioNTT2022Encoder(nn.Module):
	"""
	Encoder network from BYOLA-v2
	Copy-paste from https://github.com/nttcslab/byol-a/blob/master/v2/byol_a2/models.py
	"""
	def __init__(self, n_mels=64, d=3072, base_d=64, mlp_hidden_d=2048, conv_layers=2, stack=True):
		super().__init__()
		convs = [
			nn.Conv2d(1, base_d, 3, stride=1, padding=1),
			nn.BatchNorm2d(base_d),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),
		]
		for c in range(1, conv_layers):
			convs.extend([
				nn.Conv2d(base_d, base_d, 3, stride=1, padding=1),
				nn.BatchNorm2d(base_d),
				nn.ReLU(),
				nn.MaxPool2d(2, stride=2),
			])
		self.features = nn.Sequential(*convs)
		self.conv_d = base_d * (n_mels//(2**conv_layers))
		self.fc = nn.Sequential(
			nn.Linear(self.conv_d, mlp_hidden_d),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(mlp_hidden_d, d - self.conv_d),
			nn.ReLU(),
		)
		self.stack = stack

	def forward(self, x):
		x = self.features(x)       # (batch, ch, mel, time)
		x = x.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
		B, T, D, C = x.shape
		x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
		x_fc = self.fc(x)
		x = torch.hstack([x.transpose(1,2), x_fc.transpose(1,2)]).transpose(1,2) if self.stack else x_fc
		return x


class AudioNTT2022(AudioNTT2022Encoder):
	def __init__(self, n_mels=64, d=3072, mlp_hidden_d=2048):
		super().__init__(n_mels=n_mels, d=d, mlp_hidden_d=mlp_hidden_d)
		self.embed_dim = d

	def forward(self, x):
		x = super().forward(x)
		x = mean_max_pooling(x)
		return x


def mean_max_pooling(frame_embeddings):
	assert len(frame_embeddings.shape) == 3 # Batch,Time,Dimension
	(x1, _) = torch.max(frame_embeddings, dim=1)
	x2 = torch.mean(frame_embeddings, dim=1)
	x = x1 + x2
	return x


if __name__ == "__main__":
	pass