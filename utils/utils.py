import torch
import builtins
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import sys
from tqdm import tqdm
import numpy as np
import random
from einops import rearrange

from utils.torch_mlp_clf import TorchMLPClassifier
from itertools import chain


def flatten_list(lists):
	return list(chain.from_iterable(lists))


def off_diagonal(x):
	# return a flattened view of the off-diagonal elements of a square matrix
	n, m = x.shape
	assert n == m
	return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def generate_random(l, h, p):
	if random.random() > p:
		return np.random.uniform(l, h)
	return 0

"""------------------------------------Training utils---------------------------------------"""

class MultiCropWrapper(nn.Module):
	"""
	Perform forward pass separately on each resolution input.
	The inputs corresponding to a single resolution are clubbed and single
	forward is run on the same resolution inputs. Hence we do several
	forward passes = number of different resolutions used. We then
	concatenate all the output features and run the head forward on these
	concatenated features.
	"""
	def __init__(self, backbone, head):
		super().__init__()
		self.backbone = backbone
		self.head = head

	def forward(self, x, ncrops=1, **kwargs):
		recon_loss = None
		# convert to list
		if not isinstance(x, list):
			x = [x]
		idx_crops = torch.cumsum(torch.unique_consecutive(
			torch.tensor([inp.shape[-1] for inp in x]),
			return_counts=True,
		)[1], 0)
		start_idx, output = 0, torch.empty(0).to(x[0].device)
		for end_idx in idx_crops:
			_out = self.backbone(torch.cat(x[start_idx: end_idx]), **kwargs)
			# The output is a tuple with masked recon
			if isinstance(_out, tuple):
				_recon_loss = _out[1]
				if recon_loss is None:
					recon_loss = _recon_loss
				else:
					recon_loss += _recon_loss
				_out = _out[0]
			# accumulate outputs
			output = torch.cat((output, _out))
			start_idx = end_idx
		if recon_loss is not None:
			return self.head(output, ncrops), recon_loss
		return self.head(output, ncrops)


def get_param_groups(model):
	regularized = []
	not_regularized = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		# we do not regularize biases nor Norm parameters
		if name.endswith(".bias") or len(param.shape) == 1:
			not_regularized.append(param)
		else:
			regularized.append(param)
	return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
	

class LARS(torch.optim.Optimizer):
	def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
				 weight_decay_filter=False, lars_adaptation_filter=False):
		defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
						eta=eta, weight_decay_filter=weight_decay_filter,
						lars_adaptation_filter=lars_adaptation_filter)
		super().__init__(params, defaults)


	def exclude_bias_and_norm(self, p):
		return p.ndim == 1

	@torch.no_grad()
	def step(self):
		for g in self.param_groups:
			for p in g['params']:
				dp = p.grad

				if dp is None:
					continue

				if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
					dp = dp.add(p, alpha=g['weight_decay'])

				if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
					param_norm = torch.norm(p)
					update_norm = torch.norm(dp)
					one = torch.ones_like(param_norm)
					q = torch.where(param_norm > 0.,
									torch.where(update_norm > 0,
												(g['eta'] * param_norm / update_norm), one), one)
					dp = dp.mul(q)

				param_state = self.state[p]
				if 'mu' not in param_state:
					param_state['mu'] = torch.zeros_like(p)
				mu = param_state['mu']
				mu.mul_(g['momentum']).add_(dp)

				p.add_(mu, alpha=-g['lr'])


@torch.no_grad()
def eval_knn(net, memory_data_loader, test_data_loader, epoch, epochs, c, k=200, temperature=0.5):
	net.eval()
	total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
	# generate feature bank and target bank
	for data_tuple in tqdm(memory_data_loader, desc='Feature extracting'):
		(data, _), target = data_tuple
		target_bank.append(target)
		feature = net(data.cuda(non_blocking=True))
		feature_bank.append(feature)
	# [D, N]
	feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
	# [N]
	feature_labels = torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
	# loop test data to predict the label by weighted knn search
	test_bar = tqdm(test_data_loader)
	for data_tuple in test_bar:
		(data, _), target = data_tuple
		data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
		feature, out = net(data)

		total_num += data.size(0)
		# compute cos similarity between each feature vector and feature bank ---> [B, N]
		sim_matrix = torch.mm(feature, feature_bank)
		# [B, K]
		sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
		# [B, K]
		sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
		sim_weight = (sim_weight / temperature).exp()

		# counts for each class
		one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
		# [B*K, C]
		one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
		# weighted score ---> [B, C]
		pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

		pred_labels = pred_scores.argsort(dim=-1, descending=True)
		total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
		total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
		test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
									.format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

	return total_top1 / total_num * 100, total_top5 / total_num * 100


def eval_linear_low_shot(X_train, y_train, X_val, y_val, X_test, y_test, n):

	subset_1, subset_2, subset_3 = {}, {}, {}
	for idx, label in enumerate(y_train):
		classes = np.nonzero(label)[0]
		for c in classes:
			subset_1.setdefault(c, [])
			subset_2.setdefault(c, [])
			subset_3.setdefault(c, [])

			if len(subset_1[c]) < n:
				subset_1[c].append(idx)
			elif len(subset_2[c]) < n:
				subset_2[c].append(idx)
			elif len(subset_3[c]) < n:
				subset_3[c].append(idx)

	subset_1 = np.unique(flatten_list([idxs for idxs in subset_1.values()]))
	subset_2 = np.unique(flatten_list([idxs for idxs in subset_2.values()]))
	subset_3 = np.unique(flatten_list([idxs for idxs in subset_3.values()]))

	clf = TorchMLPClassifier(
		hidden_layer_sizes=(1024,),
		max_iter=500,
		early_stopping=True,
		n_iter_no_change=20,
		debug=True,
	)

	clf.fit(X_train[subset_1], y_train[subset_1], X_val=X_val, y_val=y_val)
	score_1 = clf.score(X_test, y_test)
	clf.fit(X_train[subset_2], y_train[subset_2], X_val=X_val, y_val=y_val)
	score_2 = clf.score(X_test, y_test)
	clf.fit(X_train[subset_3], y_train[subset_3], X_val=X_val, y_val=y_val)
	score_3 = clf.score(X_test, y_test)

	scores = [score_1, score_2, score_3]
	return np.mean(scores), np.std(scores)


def encode_vit(model, x, split_frames=True, use_cls=True):
	patch_fbins = model.grid_size()[0]
	embed_d = model.embed_dim
	unit_frames = model.img_size[1]  # number of time frames for inputs 
	# pad input's (x's) number of frames so that it's an integer multiple of unit_frames
	pad_frames = unit_frames - (x.shape[-1] % unit_frames)
	if pad_frames > 0:
		x = F.pad(x, (0, pad_frames))

	if split_frames:  # process each unit frame separately
		embeddings = []
		if use_cls:
			# [CLS] embeddings only
			for i in range(x.shape[-1] // unit_frames):
				emb = model(x[..., i*unit_frames:(i+1)*unit_frames])
				emb = emb.unsqueeze(1)  # [emb] = [b, 1, d]
				embeddings.append(emb)

			# concat along the 2nd dimension (dim=1), i.e., concat. [CLS] tokens from the different divided segments
			x = torch.hstack(embeddings)  # [x] = [b, n_unit_frames, d], n_unit_frames = x.shape[-1] // unit_frames
		else:
			# stack embeddings
			for i in range(x.shape[-1] // unit_frames):
				emb = model(x[..., i*unit_frames:(i+1)*unit_frames], return_all=True)
				emb = emb[:, 1:, :]
				emb = rearrange(emb, ' b (f t) d -> b t (f d)', f=patch_fbins, d=embed_d)
				embeddings.append(emb)
			# concat along the 2nd dimension (dim=1)
			x = torch.hstack(embeddings)  # [x] = [b, n_unit_frames*patch_tbins, patch_fbins*d]
			pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
			if pad_emb_frames > 0:
				x = x[:, :-pad_emb_frames]  # remove padded tails
		x = torch.mean(x, dim=1)  # [x] = [b, d]
	else:
		x = model(x)  # [x] = [b, d]

	return x 


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
				

"""--------------------------------For distributed training---------------------------------"""
def init_distributed_mode(cfg):

	if 'RANK' in os.environ:
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
	elif torch.cuda.is_available():
		print('Will run the code on one GPU.')
		cfg.rank, cfg.gpu, cfg.world_size = 0, 0, 1
	else:
		print('Does not support training without GPU.')
		sys.exit(1)
		
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


def model_setup_ddp(gpu, model):
	model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model = nn.parallel.DistributedDataParallel(
		model,
		device_ids=[gpu],
		output_device=gpu,
	)
	return model.module

