"""
A PyTorch implementation of Vision Transformers [Dosovitskiy et al., 2020],
adapted for our purposes (appropriate for spectrogram input, w/ masking).
References:
	https://github.com/facebookresearch/mae/blob/main/models_mae.py
	https://github.com/facebookresearch/msn/blob/main/src/deit.py
	https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
	https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py
	https://github.com/YuanGongND/ssast/blob/main/src/models/ast_models.py#L76
"""
import torch
import torch.nn as nn

import timm
from timm.models.vision_transformer import PatchEmbed, Block

from functools import partial


class MaskedSpectrogramTransformer(nn.Module):
    def __init__(
        img_size=(64, 96),
		patch_size=(16, 16),
		in_chans=1,
		embed_dim=768,
		depth=12,
		num_heads=12,
		mlp_ratio=4.,
		norm_layer=nn.LayerNorm,
	):
        super().__init__()

		# projects input to have dim. embed_dim (conv2d layer, kernel_size=patch_size, stride=patch_size)
		self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
		num_patches = self.patch_embed.num_patches

		# [CLS] token
		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		# (learned) positional embeddings
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))

		self.blocks = nn.ModuleList([
			Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
			      qk_scale=None, norm_layer=norm_layer)
			for i in range(depth)])

		self.norm = norm_layer(embed_dim)

		# timm's trunc_normal_(std=.02) is effectively normal_(std=.02) as cutoff is too big (2.)
        	torch.nn.init.normal_(self.cls_token, std=.02)
		torch.nn.init.normal_(self.pos_embed, std=.02)
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
        ids_keep = ids_shuffle[:,:len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
       	mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
		
    
    
    def forward(self, imgs, mask_ratio=0.75):
        # patchify + embed imgs
	x = self.patch_embed(imgs)
		
	# add pos embed w/o cls token
	x = x + self.pos_embed[:,1:,:]
		
	# masking: length -> length * mask_ratio
	x, mask, ids_restore = self.random_masking(x, mask_ratio)
		
	# append cls token
	cls_token = self.cls_token + self.pos_embed[:,:1,:]
	cls_token = cls_token.expand(x.shape[0], -1, -1)  # expand dim 0 of cls token to equal batch size
	x = torch.cat((cls_tokens, x), dim=1)
		
	# apply Transformer blocks
	for blk in self.blocks:
	    x = blk(x)
	x = self.norm(x)
		
	# return cls token global clip representation
	x = x[:, 0]
		
	return x, mask, ids_restore
		
	



def mst_vit_base_p16x16(patch_size=(16,16), **kwargs):
    model = MaskedSpectrogramTransformer(
	patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
	qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
