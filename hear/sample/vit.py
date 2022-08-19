"""
HEAR Benchmark submission module for (vision) transformer convolutional encoder,  
following the common API as detailed at: https://hearbenchmark.com/hear-api.htmlguidelines
References:
	ttps://github.com/nttcslab/msm-mae/blob/main/hear/hear_msm/sample.py
"""
from typing import List, Tuple
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchaudio.transforms import MelSpectrogram

from models import mae
import hear.utils as utils


def get_to_melspec(cfg):
	to_melspec = MelSpectrogram(
		sample_rate=cfg.sample_rate,
		n_fft=cfg.n_fft,
		win_length=cfg.win_length,
		hop_length=cfg.hop_length,
		n_mels=cfg.n_mels,
		f_min=cfg.f_min,
		f_max=cfg.f_max,
		power=2,
	)
	return to_melspec


class ViTModelWrapper(nn.Module):
	def __init__(self, cfg, model_type, model_file_path, patch_size):
		super().__init__()
		# needed for HEAR API
		self.cfg = cfg
		self.use_cls = True if self.cfg.use_cls is None else self.cfg.use_cls
		self.sample_rate = cfg.sample_rate
		embed_size = self._get_model(model_type, patch_size)
		self._load_weights(model_file_path)
		self.scene_embedding_size = embed_size
		self.timestamp_embedding_size = embed_size * self.model.grid_size()[0]
		self.to_melspec = get_to_melspec(cfg)


	def _get_model(self, model_type, patch_size):

		c = "vitc" in model_type
		size = model_type.split('_')[-1]
		self.model = mae.get_mae_vit(size, patch_size, c)
		return self.model.embed_dim


	def _load_weights(self, model_file_path):
		sd = torch.load(model_file_path, map_location='cpu')
		sd = sd.get('model')
		sd = {k.replace("backbone.encoder.encoder.", ""): v for k, v in sd.items() if "backbone.encoder.encoder." in k}
		self.model.load_state_dict(sd, strict=True)

	
	def _get_timestamps(self, batch_audio, x):
		audio_len = len(batch_audio[0])
		sec = audio_len / self.cfg.sample_rate
		x_len = len(x[0])
		step = sec / x_len
		ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
		ts = ts.repeat(len(batch_audio), 1)
		return ts


	def _to_feature(self, batch_audio):
		x = self.to_melspec(batch_audio)
		x = (x + torch.finfo().eps).log()
		x = x.unsqueeze(1)
		return x
	

	def _normalize_batch(self, x):
		mu, sigma = x.mean(), x.std()
		x = (x - mu) / sigma
		return x


	def _to_normalized_spec(self, batch_audio):
		x = self._to_feature(batch_audio)
		x = self._normalize_batch(x)
		return x 

	
	def _encode_lms(self, x, cls_only=None):

		if cls_only is None:
			cls_only = self.use_cls

		patch_fbins = self.model.grid_size()[0]
		embed_d = self.model.embed_dim
		unit_frames = self.model.img_size[1]  # number of time frames for inputs 
		# pad input's (x's) number of frames so that it's an integer multiple of unit_frames
		pad_frames = unit_frames - (x.shape[-1] % unit_frames)
		if pad_frames > 0:
			x = F.pad(x, (0, pad_frames))

		embeddings = []
		if cls_only:
			# [CLS] embeddings only
			for i in range(x.shape[-1] // unit_frames):
				emb = self.model(x[..., i*unit_frames:(i+1)*unit_frames])
				emb = emb.unsqueeze(1)  # [emb] = [b, 1, d]
				embeddings.append(emb)

			# concat along the 2nd dimension (dim=1), i.e., concat. [CLS] tokens from the different divided segments
			x = torch.hstack(embeddings)  # [x] = [b, n_unit_frames, d], n_unit_frames = x.shape[-1] // unit_frames
		else:
			# stack embeddings
			for i in range(x.shape[-1] // unit_frames):
				emb = self.model(x[..., i*unit_frames:(i+1)*unit_frames], return_all=True)
				emb = emb[:, 1:, :]
				emb = rearrange(emb, ' b (f t) d -> b t (f d)', f=patch_fbins, d=embed_d)
				embeddings.append(emb)
			# concat along the 2nd dimension (dim=1)
			x = torch.hstack(embeddings)  # [x] = [b, n_unit_frames*patch_tbins, patch_fbins*d]
			pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
			if pad_emb_frames > 0:
				x = x[:, :-pad_emb_frames]  # remove padded tails
		return x 
	
	
	def _encode(self, batch_audio):
		x = self._to_normalized_spec(batch_audio)
		return self._encode_lms(x)
	

	def get_scene_embeddings(self, audio):
		"""
		audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
		Returns:
			embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
		"""
		x = self._encode(audio)
		x = torch.mean(x, dim=1)  # average [CLS] tokens from different audio segments (orig. clip split into lengths of cfg.input_size[1] = 96) 
		return x


	def get_timestamp_embeddings(self, audio):
		"""
		audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
		Returns:
			embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
			timestamps: A float32 Tensor with shape (n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
		"""
		x = self._encode(audio, cls_only=False)
		ts = self._get_timestamps(audio, x)
		return x, ts 


def load_model(model_file_path: str = "", model_type: str = "vitc_base", patch_size: str = "16x8", cfg_path: str = "hear/config.yaml") -> torch.nn.Module:
	"""Load pre-trained model
	Parameters
	----------
	model_name: str, the name for pretrained model
	model_file_path: str, the path for pretrained model
	cfg_path: str, the path for yaml file including parameters value
	Returns
	-------
	torch.nn.Module object 
		Model loaded with pre-training weights
	"""
	# Load config file
	cfg = utils.load_yaml_config(cfg_path)
	# Convert patch size string to list
	patch_size = [int(patch_size.split("x")[0]), int(patch_size.split("x")[-1])]
	# Load pretrained weights
	model = ViTModelWrapper(cfg, model_type, model_file_path, patch_size)
	if torch.cuda.is_available():
		model.cuda()
	return model


def get_timestamp_embeddings(
	audio_list: List,
	model: torch.nn.Module,
) -> Tuple[Tensor, Tensor]:
	"""
	This function returns embeddings at regular intervals centered at timestamps. Both
	the embeddings and corresponding timestamps (in milliseconds) are returned.
	Args:
		audio_list: List of torch tensor audios.
		model: Loaded model.
	Returns:
		- Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
			model.timestamp_embedding_size).
		- Tensor: timestamps, Centered timestamps in milliseconds corresponding
			to each embedding in the output. Shape: (n_sounds, n_timestamps).
	"""

	# Send the model to the same device that the audio tensor is on.
	model = model.to(audio_list[0].device)
	model.eval()
	with torch.no_grad():
		return model.get_timestamp_embeddings(audio_list)


def get_scene_embeddings(
	audio_list: List,
	model: torch.nn.Module,
) -> Tensor:
	"""
	This function returns a single embedding for each audio clip. 
	Args:
		audio_list: list of torch tensor audios (audios should be resampled to 16kHz).
		model: Loaded model.
	Returns:
		- embeddings, A float32 Tensor with shape
			(n_sounds, model.scene_embedding_size).
	"""
	# Check if device has cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	model.eval()
	with torch.no_grad():
		return model.get_scene_embeddings(audio_list)