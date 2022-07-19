"""
Mask random patches in input lms (fraction to mask controlled by parameter 'mask_ratio').
Only valid approach when using a convolutional encoder (e.g., resnet). 
Code adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py.
"""

import torch 
import torch.nn as nn


def random_masking_conv(x, mask_ratio=0.75, patch_size=(16,16)):
	"""
	Perform per-sample random masking of patches on input lms by per-sample random shuffling.
	Per-sample shuffling is done by argsort random noise.
	
	Args:
		x (torch.tensor): [N, 1, F, T] (batch_size, 1, n_mels, n_frames)
		mask_ratio (float): fraction of patches to mask 
		patch_size (float): size of patches (h, w)
		
	Returns:
		masked_x (torch.tensor) : [N, 1, F, T], output lms with randomly masked patches (set to 0)
		mask (torch.tensor) : [N, 1, F, T], mask locations (1 == no mask, 0 == mask)
	"""
	
	N, _, F, T = x.shape
	
	# extract patches
	patches = nn.functional.unfold(x, kernel_size=patch_size, stride=patch_size)
	# [patches] = [N, D, L], where D = h*w, (h, w) = patch_size, L = (F*T)/(h*w)
	patches = patches.permute(0, 2, 1)
	# [patches] = [N, L, D]
	_, L, D = patches.shape
	
	len_keep = int(L * (1 - mask_ratio))
	# generate noise for random masking
	noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
	
	# sort noise for each sample
	ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
	ids_restore = torch.argsort(ids_shuffle, dim=1)
	
	# generate the binary mask: 0 is remove, 1 is keep
	mask = torch.zeros([N, L], device=x.device)
	mask[:, :len_keep] = 1
	# unshuffle to get the binary mask
	mask = torch.gather(mask, dim=1, index=ids_restore)
	mask = mask.unsqueeze(-1).repeat(1,1,D)  # [mask] = [patches] = [N, L, D]
	
	# apply mask to patches
	masked_patches = patches * mask
	# reshape masked_patches: [N, L, D] -> [N, D, L]
	masked_patches = masked_patches.permute(0, 2, 1)
	
	# refold 
	masked_x = nn.functional.fold(masked_patches, output_size=x.shape[2:], kernel_size=patch_size, stride=patch_size)
	# [masked_x] = [x] = [N, 1, F, T]
	mask = nn.functional.fold(mask.permute(0, 2, 1), output_size=x.shape[2:], kernel_size=patch_size, stride=patch_size)
	
	return masked_x, mask