import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from models import resnet, mae
from utils import utils

off_diagonal = utils.off_diagonal

class BarlowTwinsHead(nn.Module):
	def __init__(self, cfg, in_dim):
		super().__init__()
		self.cfg = cfg

		sizes = [in_dim] + self.cfg.projector_n_hidden_layers*[self.cfg.projector_hidden_dim] + [self.cfg.projector_out_dim]
		layers = []
		for i in range(len(sizes) - 2):
			layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
			layers.append(nn.BatchNorm1d(sizes[i + 1]))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
		self.projector = nn.Sequential(*layers)

		self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
	
	def forward(self, x, ncrops=1):
		x_crops = x.chunk(ncrops)
		z = torch.empty(0).to(x_crops[0].device)
		for _x in x_crops:
			_z = self.bn(self.projector(_x))
			z = torch.cat((z, _z))
		return z
		

class ModelWrapper(nn.Module):
	
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._setup_model()
	
	def _setup_model(self):
		
		if self.cfg.model_type == 'resnet50':
			self.encoder = resnet.resnet50()
			self.encoder.fc = nn.Identity()
			self.encoder.embed_dim = 2048
		elif self.cfg.model_type == 'resnet50_ReGP_NRF':
			self.encoder = resnet.resnet50_ReGP_NRF()
			self.encoder.fc = nn.Identity()
			self.encoder.embed_dim = 16384
		elif self.cfg.model_type == 'audiontt':
			assert self.cfg.n_mels == 64, f'n_mels must be 64 to use AudioNTT encoder (n_mels set to {self.cfg.n_mels})'
			self.encoder = AudioNTT2022(squeeze_excitation=self.cfg.squeeze_excitation)
		elif 'vit' in self.cfg.model_type:
			conv_stem_bool = self.cfg.model_type.split('_')[0] == 'vitc'
			self.encoder = ViT( 
				dataset=self.cfg.dataset,
				size=self.cfg.model_type.split('_')[-1], 
				patch_size=self.cfg.patch_size,
				c=conv_stem_bool,
				use_learned_pos_embd=self.cfg.use_learned_pos_embd,
				use_mean_pool=self.cfg.use_mean_pool,
				use_decoder=self.cfg.masked_recon,
			)
		else:
			raise NotImplementedError(f'Model type {self.cfg.model_type} is not supported')
		self.feature_dim = self.encoder.embed_dim

	def forward(self, x, mask_ratio=0, masked_recon=False):
		if 'vit' in self.cfg.model_type:
			return self.encoder(x, mask_ratio=mask_ratio, masked_recon=masked_recon)
		return self.encoder(x)



class ViT(nn.Module):
	def __init__(self, dataset='fsd50k', size='base', patch_size=None, c=True,
			use_learned_pos_embd=False, use_mean_pool=False, use_decoder=False):
		super().__init__()
		
		if patch_size is None:
			patch_size = [16, 16]
		if dataset == 'cifar10':
			self.encoder = mae.get_mae_vit(size, patch_size, c, use_learned_pos_embd=use_learned_pos_embd,
										img_size=(32,32), in_chans=3)
		else:
			self.encoder = mae.get_mae_vit(size, patch_size, c, use_learned_pos_embd=use_learned_pos_embd,
										use_decoder=use_decoder)
		
		self.embed_dim = self.encoder.embed_dim
		self.use_mean_pool = use_mean_pool

	def forward(self, x, mask_ratio=0, masked_recon=False):
		x = self.encoder(x, mask_ratio=mask_ratio, masked_recon=masked_recon,
					mean_pool=self.use_mean_pool)
		return x


class AudioNTT2022Encoder(nn.Module):
	"""
	Encoder network from BYOLA-v2
	Copy-paste from https://github.com/nttcslab/byol-a/blob/master/v2/byol_a2/models.py
	"""
	def __init__(self, n_mels=64, d=3072, base_d=64, mlp_hidden_d=2048, conv_layers=2, stack=True, squeeze_excitation=False):
		super().__init__()
		convs = [
			nn.Conv2d(1, base_d, 3, stride=1, padding=1),
			nn.BatchNorm2d(base_d),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),
		]
		if squeeze_excitation:
			convs.append(SE_Block(c=base_d))
		for c in range(1, conv_layers):
			convs.extend([
				nn.Conv2d(base_d, base_d, 3, stride=1, padding=1),
				nn.BatchNorm2d(base_d),
				nn.ReLU(),
				nn.MaxPool2d(2, stride=2),
			])
			if squeeze_excitation:
				convs.append(SE_Block(c=base_d))
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
	def __init__(self, n_mels=64, d=3072, mlp_hidden_d=2048, squeeze_excitation=False):
		super().__init__(n_mels=n_mels, d=d, mlp_hidden_d=mlp_hidden_d, squeeze_excitation=squeeze_excitation)
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


class SE_Block(nn.Module):
    """Copy-paste from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4 """
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)



"""
BYOL-like asymmetric learning updates (with Barlow Twins loss)
Ref: https://github.com/lucidrains/byol-pytorch/
"""

class EMA():
	def __init__(self, beta):
		super().__init__()
		self.beta = beta

	def update_average(self, old, new):
		if old is None:
			return new
		return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
	for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
		old_weight, up_weight = ma_params.data, current_params.data
		ma_params.data = ema_updater.update_average(old_weight, up_weight)


class BarlowTwinsBYOL(nn.Module):
	
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._setup_model()
	

	def _setup_model(self):
		
		if self.cfg.model_type == 'resnet50':
			self.encoder = resnet.resnet50()
			self.encoder.fc = nn.Identity()
			self.encoder.embed_dim = 2048
		elif self.cfg.model_type == 'resnet50_ReGP_NRF':
			self.encoder = resnet.resnet50_ReGP_NRF()
			self.encoder.fc = nn.Identity()
			self.encoder.embed_dim = 16384
		elif self.cfg.model_type == 'audiontt':
			self.encoder = AudioNTT2022()
		elif 'vit' in self.cfg.model_type:
			if self.cfg.model_type.split('_')[0] == 'vitc':
				self.encoder = ViT(c=True, size=self.cfg.model_type.split('_')[-1])
			else:
				self.encoder = ViT(c=False, size=self.cfg.model_type.split('_')[-1])
		else:
			raise NotImplementedError(f'Model type {self.cfg.model_type} is not supported')
		feature_dim = self.encoder.embed_dim

		# target encoder
		self.target_encoder = copy.deepcopy(self.encoder)
		if self.cfg.stop_gradient:
			for p in self.target_encoder.parameters():
				p.requires_grad = False
		self.target_ema_updater = EMA(self.cfg.moving_average_decay)
		
		sizes = [feature_dim] + self.cfg.projector_n_hidden_layers*[self.cfg.projector_hidden_dim] + [self.cfg.projector_out_dim]
		layers = []
		for i in range(len(sizes) - 2):
			layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
			layers.append(nn.BatchNorm1d(sizes[i + 1]))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
		self.projector = nn.Sequential(*layers)

		# predictor 
		if self.cfg.predictor:
			self.online_predictor = nn.Sequential(
				nn.Linear(sizes[-1], sizes[-1], bias=False),
				nn.BatchNorm1d(sizes[-1]),
				nn.ReLU(inplace=True),
				nn.Linear(sizes[-1], sizes[-1], bias=False),
			)
		
		self.bn = nn.BatchNorm1d(sizes[-1], affine=False)


	def update_moving_average(self):
		update_moving_average(self.target_ema_updater, self.target_encoder, self.encoder)
	

	def forward(self, y1, y2):
		
		feature1 = self.encoder(y1)
		feature2 = self.encoder(y2)

		online_z1 = self.projector(feature1)
		online_z2 = self.projector(feature2)

		if self.cfg.predictor:
			online_z1 = self.online_predictor(online_z1)
			online_z2 = self.online_predictor(online_z2)
		
		if self.cfg.stop_gradient:
			with torch.no_grad():
				target_z1 = self.projector(self.target_encoder(y1))
				target_z2 = self.projector(self.target_encoder(y2))
				target_z1.detach_()
				target_z2.detach_()
		else:
			target_z1 = self.projector(self.target_encoder(y1))
			target_z2 = self.projector(self.target_encoder(y2))

		
		# empirical cross-correlation matrix
		c1 = self.bn(online_z1).T @ self.bn(target_z2)
		c2 = self.bn(online_z2).T @ self.bn(target_z1)
		
		# sum the cross-correlation matrix between all gpus
		c1.div_(online_z1.shape[0])
		c2.div_(online_z1.shape[0])
		if utils.is_dist_avail_and_initialized():
			torch.distributed.all_reduce(c1)
			torch.distributed.all_reduce(c2)
		
		on_diag_1 = torch.diagonal(c1).add_(-1).pow_(2).sum()
		on_diag_2 = torch.diagonal(c2).add_(-1).pow_(2).sum()
		if self.cfg.HSIC:
			# encouraging off_diag to be negative ones
			off_diag_1 = off_diagonal(c1).add_(1).pow_(2).sum()
			off_diag_2 = off_diagonal(c2).add_(1).pow_(2).sum()
		else:
			off_diag_1 = off_diagonal(c1).pow_(2).sum()
			off_diag_2 = off_diagonal(c2).pow_(2).sum()

		loss_1 = on_diag_1 + self.cfg.lmbda * off_diag_1
		loss_2 = on_diag_2 + self.cfg.lmbda * off_diag_2
		loss = loss_1 + loss_2
		return loss.mean()



if __name__ == "__main__":
	pass
