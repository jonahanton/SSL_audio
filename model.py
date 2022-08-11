import torch
import torch.nn as nn
import torch.nn.functional as F

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
	
	def forward(self, x, ncrops=2):
		x_crops = x.chunk(ncrops)
		z = torch.empty(0).to(x_crops[0].device)
		for _x in x_crops:
			_z = self.projector(_x)
			z = torch.cat((z, _z))
		return z


class BarlowTwinsPredictor(nn.Module):
	def __init__(self, in_dim, use=True):
		super().__init__()

		self.predictor = nn.Identity()
		if use:
			self.predictor = nn.Sequential(
				nn.Linear(in_dim, in_dim, bias=False),
				nn.BatchNorm1d(in_dim),
				nn.ReLU(inplace=True),
				nn.Linear(in_dim, in_dim, bias=False),
			)
	
	def forward(self, x, ncrops=2):
		x_crops = x.chunk(ncrops)
		z = torch.empty(0).to(x_crops[0].device)
		for _x in x_crops:
			_z = self.predictor(_x)
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
		elif self.cfg.model_type == 'resnet18':
			self.encoder = resnet.resnet18()
			self.encoder.fc = nn.Identity()
			self.encoder.embed_dim = 512
		elif self.cfg.model_type == 'resnet50_ReGP_NRF':
			self.encoder = resnet.resnet18_ReGP_NRF()
			self.encoder.fc = nn.Identity()
			self.encoder.embed_dim = 4096
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


if __name__ == "__main__":
	pass
