import torch
import builtins
import datetime
import torch.distributed as dist
import os
import sys


"""------------------------------------Training utils---------------------------------------"""

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


