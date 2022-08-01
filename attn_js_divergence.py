import argparse

import torch 
import torch.nn as nn
from scipy.spatial.distance import jensenshannon

from models.mae import mae


MODELS = [
	'vit_base', 'vit_small', 'vit_tiny',
	'vitc_base', 'vitc_small', 'vitc_tiny',
]


def get_std_parser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--unit_sec', type=float, default=0.95)
	parser.add_argument('--crop_frames', type=int, default=96)
	parser.add_argument('--sample_rate', type=int, default=16000)
	parser.add_argument('--n_fft', type=int, default=1024)
	parser.add_argument('--win_length', type=int, default=1024)
	parser.add_argument('--hop_length', type=int, default=160)
	parser.add_argument('--n_mels', type=int, default=64)
	parser.add_argument('--f_min', type=int, default=60)
	parser.add_argument('--f_max', type=int, default=7800)
	parser.add_argument('--use_learned_pos_embd', action='store_true', default=False)
	return parser


class AttnModelWrapper(nn.Module):

	def __init__(self, cfg):
		self.cfg = cfg
		self._get_model(cfg.model_type)
		self._load_weights(cfg.model_file_path)

	def _get_model(self, model_type):
		c = "vitc" in model_type
		size = model_type.split('_')[-1]
		if c:
			if size == 'base':
				self.model = mae.mae_vitc_base_patch16x16()
			elif size == 'small':
				self.model = mae.mae_vitc_small_patch16x16()
			elif size == 'tiny':
				self.model = mae.mae_vitc_tiny_patch16x16()
			else:
				raise NotImplementedError(f'ViTc size {size} is not supported')
		else:
			if size == 'base':
				self.model = mae.mae_vit_base_patch16x16()
			elif size == 'small':
				self.model = mae.mae_vit_small_patch16x16()
			elif size == 'tiny':
				self.model = mae.mae_vit_tiny_patch16x16()
			else:
				raise NotImplementedError(f'ViT size {size} is not supported')

	def _load_weights(self, model_file_path):
		if model_file_path is None:
			return
		sd = torch.load(model_file_path, map_location='cpu')
		sd = {k.replace("encoder.encoder.", ""): v for k, v in sd.items() if "encoder.encoder." in k}
		self.model.load_state_dict(sd, strict=True)

	def forward(self, x):
		return self.model.forward_attn(x)

	def forward_js_divergence(self, x):
		# x = [B, 1, F, T]
		attns = self(x)  # [n_layers, B, n_heads, n_tokens, n_tokens]
		n_l, B, n_h, n_t, _ = attns.shape
		attns = torch.flatten(attns, start_dim=3)  # [n_layers, B, n_heads, n_tokens**2]
		attns = attns.permute(2, 0, 1, 3)  # [B, n_heads, n_layers, n_tokens**2]
		js = torch.zeros((b, n_h, n_l - 1, n_l - 1))
		for b in range(B):
			for h in range(n_h):
				for l1 in range(n_l):
					for l2 in range(n_l):
						if l1 == l2:
							continue 
						attn_vec1 = attns[b, h, l1].cpu().numpy()
						attn_vec2 = attns[b, h, l2].cpu().numpy()
						js[b, h, l1, l2] = jensenshannon(attn_vec1, attn_vec2)
		




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='JS divergence for attention', parents=[get_std_parser()])
	parser.add_argument('--model_type', type=str, default='vitc_base', choices=MODELS)
	parser.add_argument('--model_file_path', type=str, default=None)
	args = parser.parse_args()

	