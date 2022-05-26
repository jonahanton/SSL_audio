"""
Implementation of Barlow Twins [Zbontar et al., 2021], 
adapted from
	https://github.com/MaxLikesMath/Barlow-Twins-Pytorch/blob/main/Twins/barlow.py
	https://github.com/facebookresearch/barlowtwins
using some code from
	https://github.com/lucidrains/byol-pytorch

"""

import torch
import torch.nn as nn


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
	
	def __init__(self, backbone, projection_sizes, lambd, mask_ratio):
		
		super().__init__()
		self.backbone = backbone
		self.lambd = lambd
		self.mask_ratio = mask_ratio
		
		# projector
		sizes = projection_sizes
		layers = []
		for i in range(len(sizes) - 2):
			layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
			layers.append(nn.BatchNorm1d(sizes[i + 1]))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
		self.projector = nn.Sequential(*layers)
			
		# normalization layer for the representations z1 and z2
		self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
		
	
	def forward(self, y1, y2):
		z1 = self.projector(self.backbone(y1, mask_ratio=0.))
		z2 = self.projector(self.backbone(y2, mask_ratio=self.mask_ratio))
		
		# empirical cross-correlation matrix
		c = self.bn(z1).T @ self.bn(z2)
		
		# sum the cross-correlation matrix between all gpus
		c.div_(z1.shape[0])
		torch.distributed.all_reduce(c)
		
		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		off_diag = off_diagonal(c).pow_(2).sum()
		loss = on_diag + self.lambd * off_diag
		return loss
