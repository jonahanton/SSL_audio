import torch
import builtins
import datetime
import torch.distributed as dist
import os
import sys
from tqdm import tqdm


"""------------------------------------Training utils---------------------------------------"""

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
		if isinstance(feature, list):
			feature = feature[-1]
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


