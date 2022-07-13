"""
A PyTorch implementation of Vision Transformers [Dosovitskiy et al., 2020],
adapted for our purposes (appropriate for spectrogram input, w/ masking).
Code adapted from: https://github.com/nttcslab/msm-mae

Other references:
	https://github.com/nttcslab/msm-mae
	https://github.com/facebookresearch/mae/blob/main/models_mae.py
	https://github.com/facebookresearch/msn/blob/main/src/deit.py
	https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
	https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py
	https://github.com/YuanGongND/ssast/blob/main/src/models/ast_models.py#L76
"""
from functools import partial

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp
from utils.pos_embed import get_2d_sincos_pos_embed, get_sinusoid_encoding_table


class AttentionKBiasZero(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
		super().__init__()
		assert dim % num_heads == 0, 'dim should be divisible by num_heads'
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=False)
		if qkv_bias:
			self.q_bias = nn.Parameter(torch.zeros(dim))
			self.v_bias = nn.Parameter(torch.zeros(dim))
		else:
			self.q_bias = None
			self.v_bias = None

		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		B, N, C = x.shape

		qkv_bias = None
		if self.q_bias is not None:
			qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
		qkv = nn.functional.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)

		qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class BlockKBiasZero(nn.Module):

	def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
				 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = AttentionKBiasZero(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
		# NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

	def forward(self, x, return_attention=False):
		if return_attention:
			x_att, attn = self.attn(self.norm1(x), return_attention=True)
			x = x + self.drop_path(x_att)
			x = x + self.drop_path(self.mlp(self.norm2(x)))
			return x, attn
		x = x + self.drop_path(self.attn(self.norm1(x)))
		x = x + self.drop_path(self.mlp(self.norm2(x)))
		return x


class MaskedAutoencoderViT(nn.Module):
	""" Masked Autoencoder with VisionTransformer backbone
	"""
	def __init__(self, img_size=(64, 96), patch_size=(16, 16), in_chans=1,
				 embed_dim=768, depth=12, num_heads=12,
				 use_decoder=False,
				 decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=12,
				 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
				 use_cls_token=False, block_cls=BlockKBiasZero, use_2d_dec_pos_embd=False):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim
		self.use_cls_token = use_cls_token
		self.use_decoder = use_decoder

		# --------------------------------------------------------------------------
		# MAE encoder specifics
		self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
		num_patches = self.patch_embed.num_patches

		total_patches = num_patches + (1 if use_cls_token else 0)
		if use_cls_token:
			self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		else:
			print('NO [CLS] TOKEN')
		self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

		self.blocks = nn.ModuleList([
			block_cls(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
			for i in range(depth)])
		self.norm = norm_layer(embed_dim)
		# --------------------------------------------------------------------------

		# --------------------------------------------------------------------------
		# MAE decoder specifics
		if use_decoder:
			self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

			self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

			self.decoder_pos_embed = nn.Parameter(torch.zeros(1, total_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

			self.decoder_blocks = nn.ModuleList([
				block_cls(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
				for i in range(decoder_depth)])

			self.decoder_norm = norm_layer(decoder_embed_dim)
			self.decoder_pred = nn.Linear(decoder_embed_dim, self.img_patch_dim(), bias=True) # decoder to patch
		# --------------------------------------------------------------------------

		self.norm_pix_loss = norm_pix_loss

		self.initialize_weights(use_2d_dec_pos_embd)

		print(f'{self.__class__.__name__}(patch size={self.patch_size()}, grid_size={self.grid_size()},\n'
			  f'  embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, decoder_embed_dim={decoder_embed_dim},\n'
			  f'  use_decoder={use_decoder}, decoder_depth={decoder_depth}, decoder_num_heads={decoder_num_heads}, mlp_ratio={mlp_ratio},\n'
			  f'  norm_pix_loss={norm_pix_loss}, use_cls_token={use_cls_token}, use_2d_dec_pos_embd={use_2d_dec_pos_embd})')

	def patch_size(self):
		return self.patch_embed.proj.kernel_size

	def grid_size(self):
		return self.patch_embed.grid_size

	def img_patch_dim(self):
		patch_size = self.patch_size()
		return patch_size[0] * patch_size[1] * self.in_chans

	def initialize_weights(self, use_2d_dec_pos_embd=False):
		# initialization
		# initialize (and freeze) pos_embed by sin-cos embedding
		pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size(), cls_token=self.use_cls_token)
		self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

		if self.use_decoder:
			if use_2d_dec_pos_embd:
				decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size(), cls_token=self.use_cls_token)
			else:
				grid_patches = self.grid_size()[0] * self.grid_size()[1]
				decoder_pos_embed = get_sinusoid_encoding_table(grid_patches, self.decoder_pos_embed.shape[-1], cls_token=self.use_cls_token)
			self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

		# initialize patch_embed like nn.Linear (instead of nn.Conv2d)
		w = self.patch_embed.proj.weight.data # shape=torch.Size([768, 1, 16, 16])
		torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

		# timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
		if self.use_cls_token:
			torch.nn.init.normal_(self.cls_token, std=.02)
		if self.use_decoder:
			torch.nn.init.normal_(self.mask_token, std=.02)

		# initialize nn.Linear and nn.LayerNorm
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			# we use xavier_uniform following official JAX ViT:
			torch.nn.init.xavier_uniform_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def patchify(self, imgs):
		"""
		imgs: (N, C, F, T)
		x: (N, L, patch_size[0]*patch_size[0]*in_chans)
		"""

		ph, pw = self.patch_size()
		h, w = self.grid_size()
		x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, ph, w, pw))
		x = torch.einsum('nchpwq->nhwpqc', x)
		x = x.reshape(shape=(imgs.shape[0], h * w, self.img_patch_dim()))
		return x

	def unpatchify(self, x):
		"""
		x: (N, L, patch_size[0]*patch_size[0]*in_chans)
		imgs: (N, C, H, W)
		"""
		ph, pw = self.patch_size()
		h, w = self.grid_size()
		assert h * w == x.shape[1]
		
		x = x.reshape(shape=(x.shape[0], h, w, ph, pw, self.in_chans))
		x = torch.einsum('nhwpqc->nchpwq', x)
		imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * ph, w * pw))
		return imgs

	def random_masking(self, x, mask_ratio):
		"""
		Perform per-sample random masking by per-sample shuffling.
		Per-sample shuffling is done by argsort random noise.
		x: [N, L, D], sequence
		"""
		N, L, D = x.shape  # batch, length, dim

		if isinstance(mask_ratio, (torch.Tensor, np.ndarray, list, tuple)):
			# Prefixed mask
			mask = mask_ratio.clone().detach()
			#ids_shuffle = torch.where(mask.reshape(N, -1) == 0)[1].reshape(N, -1)
			ids_shuffle = torch.argsort(mask.reshape(N, -1), dim=1)
			ids_restore = torch.argsort(ids_shuffle, dim=1)
			len_keep = (mask[0] == 0).sum()
		elif mask_ratio == 0:
			# No mask
			mask = torch.zeros([N, L], device=x.device)
			ids_restore = torch.tensor(list(range(L))).to(torch.int)
			return x, mask, ids_restore
		else:
			# Random mask
			len_keep = int(L * (1 - mask_ratio))
			noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
			# sort noise for each sample
			ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
			ids_restore = torch.argsort(ids_shuffle, dim=1)

		# keep the first subset
		ids_keep = ids_shuffle[:, :len_keep]
		x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

		# generate the binary mask: 0 is keep, 1 is remove
		mask = torch.ones([N, L], device=x.device)
		mask[:, :len_keep] = 0
		# unshuffle to get the binary mask
		mask = torch.gather(mask, dim=1, index=ids_restore)

		return x_masked, mask, ids_restore

	def forward_encoder(self, x, mask_ratio):
		# embed patches
		x = self.patch_embed(x)

		# add pos embed w/o cls token
		if self.use_cls_token:
			x = x + self.pos_embed[:, 1:, :]
		else:
			x = x + self.pos_embed

		# masking: length -> length * mask_ratio
		x, mask, ids_restore = self.random_masking(x, mask_ratio)

		# append cls token
		if self.use_cls_token:
			cls_token = self.cls_token + self.pos_embed[:, :1, :]
			cls_tokens = cls_token.expand(x.shape[0], -1, -1)
			x = torch.cat((cls_tokens, x), dim=1)

		# apply Transformer blocks
		for blk in self.blocks:
			x = blk(x)
		x = self.norm(x)

		return x, mask, ids_restore

	def forward_decoder(self, x, ids_restore):
		# embed tokens
		x = self.decoder_embed(x)

		# append mask tokens to sequence
		mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
		x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
		x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
		if self.use_cls_token:
			x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
		else:
			x = x_

		# add pos embed
		x = x + self.decoder_pos_embed

		# apply Transformer blocks
		for blk in self.decoder_blocks:
			x = blk(x)
		x = self.decoder_norm(x)

		# predictor projection
		x = self.decoder_pred(x)

		# remove cls token
		if self.use_cls_token:
			x = x[:, 1:, :]

		return x

	def forward_loss(self, imgs, pred, mask):
		"""
		imgs: [N, C, H, W]
		pred: [N, L, ph*pw*C]
		mask: [N, L], 0 is keep, 1 is remove, 
		"""
		target = self.patchify(imgs)
		if self.norm_pix_loss:
			mean = target.mean(dim=-1, keepdim=True)
			var = target.var(dim=-1, keepdim=True)
			target = (target - mean) / (var + 1.e-6)**.5

		loss = (pred - target) ** 2
		loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

		loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
		return loss

	def forward(self, imgs, mask_ratio=0.75):
		latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
		# pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
		# loss = self.forward_loss(imgs, pred, mask)
		# return loss, pred, mask
		return latent, mask, ids_restore

	def forward_viz(self, imgs, mask_ratio=0.75):
		loss, pred, mask = self.forward(imgs, mask_ratio)
		# recons_as_is = self.unpatchify(pred)
		# overwrite visible patches with original image.
		pred_org_on_mask = pred.clone()
		visible = (mask == 0.)
		pred_org_on_mask[visible] = self.patchify(imgs)[visible]
		recons = self.unpatchify(pred_org_on_mask)
		errormap = ((recons - imgs) ** 2).sqrt()
		return loss, recons, errormap, mask.reshape(mask.shape[0], *self.grid_size())


def mae_vit_base_patchX(patch_size, **kwargs):
	model = MaskedAutoencoderViT(
		patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
		mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


def mae_vit_base_patch16x16(**kwargs):
	return mae_vit_base_patchX([16, 16], **kwargs)

def mae_vit_base_patch16x8(**kwargs):
	return mae_vit_base_patchX([16, 8], **kwargs)

def mae_vit_base_patch16x4(**kwargs):
	return mae_vit_base_patchX([16, 4], **kwargs)

def mae_vit_base_patch8x16(**kwargs):
	return mae_vit_base_patchX([8, 16], **kwargs)

def mae_vit_base_patch80x4(**kwargs):
	return mae_vit_base_patchX([80, 4], **kwargs)

def mae_vit_base_patch8x8(**kwargs):
	return mae_vit_base_patchX([8, 8], **kwargs)

def mae_vit_base_patch80x2(**kwargs):
	return mae_vit_base_patchX([80, 2], **kwargs)

def mae_vit_base_patch80x1(**kwargs):
	return mae_vit_base_patchX([80, 1], **kwargs)
