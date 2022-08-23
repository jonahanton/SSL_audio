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

# Default frame duration in milliseconds
TIMESTAMP_FRAME_DUR = 950
# Default hop_size in milliseconds
TIMESTAMP_HOP_SIZE = 50
# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 512


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
		if model_file_path != "":
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
		if 'model' in sd.keys():
			sd = sd.get('model')
		while True:
			clean_sd = {k.replace("backbone.encoder.encoder.", ""): v for k, v in sd.items() if "backbone.encoder.encoder." in k}
			if clean_sd:
				break
			clean_sd = {k.replace("encoder.encoder.", ""): v for k, v in sd.items() if "encoder.encoder." in k}
			if clean_sd:
				break
			clean_sd = sd
			break
		self.model.load_state_dict(clean_sd, strict=True)

	
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

	
	def encode_lms(self, x):

		unit_frames = self.model.img_size[1]  # number of time frames for inputs 
		# pad input's (x's) number of frames so that it's an integer multiple of unit_frames
		pad_frames = unit_frames - (x.shape[-1] % unit_frames)
		if pad_frames > 0:
			x = F.pad(x, (0, pad_frames))

		embeddings = []
		# [CLS] embeddings only
		for i in range(x.shape[-1] // unit_frames):
			emb = self.model(x[..., i*unit_frames:(i+1)*unit_frames])
			emb = emb.unsqueeze(1)  # [emb] = [b, 1, d]
			embeddings.append(emb)

		# concat along the 2nd dimension (dim=1), i.e., concat. [CLS] tokens from the different divided segments
		x = torch.hstack(embeddings)  # [x] = [b, n_unit_frames, d], n_unit_frames = x.shape[-1] // unit_frames
		return x 
	
	
	def encode(self, batch_audio):
		x = self._to_normalized_spec(batch_audio)
		return self.encode_lms(x)
	

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
	frame_duration: float = TIMESTAMP_FRAME_DUR,
	hop_size: float = TIMESTAMP_HOP_SIZE,
	cfg_path: str = 'hear/config.yaml'
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

	# Load config file
	cfg = utils.load_yaml_config(cfg_path)
	to_melspec = MelSpectrogram(
						sample_rate=cfg.sample_rate,
						n_fft=cfg.n_fft,
						win_length=cfg.win_length,
						hop_length=cfg.hop_length,
						n_mels=cfg.n_mels,
						f_min=cfg.f_min,
						f_max=cfg.f_max,
						).to(audio_list[0].device)

	# Send the model to the same device that the audio tensor is on.
	model = model.to(audio_list[0].device)

	# Split the input audio signals into frames and then flatten to create a tensor
	# of audio frames that can be batch processed.
	frames, timestamps = utils.frame_audio(
		audio_list,
		frame_size=int((frame_duration/1000)*cfg.sample_rate),
		hop_size=hop_size,
		sample_rate=cfg.sample_rate,
	)
	audio_batches, num_frames, _ = frames.shape
	frames = frames.flatten(end_dim=1)

	# Convert audio frames to spectrograms
	melspec_frames = ((to_melspec(frames) + torch.finfo(torch.float).eps).log())
	# Normalize 
	mean, std = utils.compute_timestamp_stats(melspec_frames)
	melspec_frames = ((melspec_frames - mean) / std).unsqueeze(0)
	melspec_frames = melspec_frames.permute(1, 0, 2, 3)
	
	# We're using a DataLoader to help with batching of frames
	dataset = torch.utils.data.TensorDataset(melspec_frames)
	loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

	# Put the model into eval mode, and not computing gradients while in inference.
	# Iterate over all batches and accumulate the embeddings for each frame.
	# Disable parameter tuning
	model.eval()
	with torch.no_grad():
		embeddings_list = [model.encode_lms(batch[0]) for batch in loader]

	# Concatenate mini-batches back together and unflatten the frames
	# to reconstruct the audio batches
	embeddings = torch.cat(embeddings_list, dim=0)
	embeddings = embeddings.unflatten(0, (audio_batches, num_frames))

	return embeddings, timestamps


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
		return torch.mean(model.encode(audio_list), dim=1)