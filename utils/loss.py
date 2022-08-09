import torch
import torch.nn as nn

import utils.utils as utils
off_diagonal = utils.off_diagonal


class BarlowTwinsLoss(nn.Module):
	def __init__(self, cfg, ncrops):
		super().__init__()
		self.cfg = cfg
		self.ncrops = ncrops
		self.bn = nn.BatchNorm1d(cfg.projector_out_dim, affine=False)

	def forward_loss(self, z1, z2):
		# empirical cross-correlation matrix
		c = self.bn(z1).T @ self.bn(z2)
		# sum the cross-correlation matrix between all gpus
		c.div_(z1.shape[0])
		if utils.is_dist_avail_and_initialized():
			torch.distributed.all_reduce(c)
		
		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		if self.cfg.HSIC:
			# encouraging off_diag to be negative ones
			off_diag = off_diagonal(c).add_(1).pow_(2).sum()
		else:
			off_diag = off_diagonal(c).pow_(2).sum()
		loss = self.cfg.alpha * on_diag + self.cfg.lmbda * off_diag
		return loss

	def forward(self, student_output, teacher_output, ngcrops_each=1):

		student_out = student_output.chunk(self.ncrops - (2-ngcrops_each))
		teacher_out = teacher_output.chunk(ngcrops_each)  # teacher only gets 1 global crop

		total_loss = 0
		n_loss_terms = 0
		for q in range(len(teacher_out)):
			for v in range(len(student_out)):
				if len(teacher_out) > 1:
					if q == v:
						continue
				loss = self.forward_loss(teacher_out[q], student_out[v])
				total_loss += loss
				n_loss_terms += 1
		total_loss /= n_loss_terms
		return total_loss