import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

from vision_transformer import mae


class ViT(nn.Module):
	def __init__(self, feature_dim=128, dataset='fsd50k', size='base', latent='cls'):
		super(ViT, self).__init__()
		self.latent = latent 

		# encoder
		if size == 'base':
			self.f = mae.mae_vit_base_patch16x16()
		embed_dim = self.f.embed_dim
		bottleneck_dim = int(embed_dim / 4)
		# projection head
		self.g = nn.Sequential(nn.Linear(embed_dim, bottleneck_dim, bias=False), nn.BatchNorm1d(bottleneck_dim),
							   nn.ReLU(inplace=True), nn.Linear(bottleneck_dim, feature_dim, bias=True))

	def forward(self, x, mask_ratio=0.):
		x = self.f(x, mask_ratio=mask_ratio)
		if self.latent == 'cls':
			x = x[:, 0]
		elif self.latent == 'pool':
			x = torch.mean(x[:, 1:], dim=1)
		feature = x.contiguous()
		out = self.g(feature)
		return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class ResNet(nn.Module):
	def __init__(self, feature_dim=128, dataset='cifar10', pretrained=False):
		super(ResNet, self).__init__()

		self.f = []
		for name, module in resnet50(pretrained=pretrained).named_children():
			if name == 'conv1':
				if dataset == 'fsd50k':
					module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
				else:
					module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
			if dataset == 'cifar10':
				if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
					self.f.append(module)
			elif dataset == 'tiny_imagenet' or dataset == 'stl10' or dataset == 'fsd50k':
				if not isinstance(module, nn.Linear):
					self.f.append(module)

		# encoder
		self.f = nn.Sequential(*self.f)
		# projection head
		self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
							   nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

	def forward(self, x, mask_ratio=0.):
		x = self.f(x)
		feature = torch.flatten(x, start_dim=1)
		out = self.g(feature)
		return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class BYOLAv2encoder(nn.Module):
	def __init__(self, feature_dim=128, dataset='fsd50k', n_mels=64, embed_dim=3072, mlp_hidden_d=2048):
		super(BYOLAv2encoder, self).__init__()

		self.f = AudioNTT2022(n_mels=n_mels, d=embed_dim, mlp_hidden_d=mlp_hidden_d)
		bottleneck_dim = int(embed_dim / 4)
		# projection head
		self.g = nn.Sequential(nn.Linear(embed_dim, bottleneck_dim, bias=False), nn.BatchNorm1d(bottleneck_dim),
							   nn.ReLU(inplace=True), nn.Linear(bottleneck_dim, feature_dim, bias=True))

	def forward(self, x, mask_ratio=0.):
		feature = self.f(x)
		out = self.g(feature)
		return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class AudioNTT2022Encoder(nn.Module):
	"""
	Encoder network taken from BYOLA-v2
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


def mean_max_pooling(frame_embeddings):
	assert len(frame_embeddings.shape) == 3 # Batch,Time,Dimension
	(x1, _) = torch.max(frame_embeddings, dim=1)
	x2 = torch.mean(frame_embeddings, dim=1)
	x = x1 + x2
	return x


class AudioNTT2022(AudioNTT2022Encoder):
	def __init__(self, n_mels=64, d=3072, mlp_hidden_d=2048):
		super().__init__(n_mels=n_mels, d=d, mlp_hidden_d=mlp_hidden_d)
		self.d = d

	def forward(self, x):
		x = super().forward(x)
		x = mean_max_pooling(x)
		return x



if __name__ == "__main__":

	model = BYOLAv2encoder(dataset='fsd50k')
	x = torch.randn(3, 1, 64, 96)
	feature, out = model(x)
	print(feature.shape)
	print(out.shape)