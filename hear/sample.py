"""
HEAR Benchmark submission module,  
following the common API as detailed at: https://hearbenchmark.com/hear-api.htmlguidelines
"""

from typing import List, Tuple
from tqdm import tqdm
from easydict import EasyDict
import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.transforms import MelSpectrogram

from model import BYOLAv2encoder, ResNet, ViT
import hear.utils as utils

# Default frame duration in milliseconds
TIMESTAMP_FRAME_DUR = 1000
# Default hop_size in milliseconds
TIMESTAMP_HOP_SIZE = 50
# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 512


class ModelWrapper(nn.Module):
	def __init__(self, cfg, model_name, model_file_path):
		super().__init__()
		# needed for HEAR API
		self.sample_rate = cfg.sample_rate
		self.model, embed_size = self._get_model(model_name)
		self._load_weights(model_file_path)
		self.scene_embedding_size = embed_size
		self.timestamp_embedding_size = embed_size

	def forward(self, x):
		return self.model(x)

	def _get_model(self, model_name):
		if model_name == 'resnet':
			return ResNet().f, 2048
		elif model_name == 'byola':
			return BYOLAv2encoder().f, 3072
		else:
			raise ValueError(f'Model {model_name} not supported!')

	def _load_weights(self, model_file_path):
		self.model.load_state_dict(torch.load(model_file_path, map_location='cpu'), strict=False)


def load_model(model_file_path: str = "", model_name: str = "resnet", cfg_path: str = "hear/config.yaml") -> torch.nn.Module:
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
	# Load pretrained weights
	model = ModelWrapper(cfg, model_name, model_file_path)
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
		frame_duration: Frame (segement) duration in milliseconds
		hop_size: Hop size in milliseconds.
			NOTE: Not required by the HEAR API. We add this optional parameter
			to improve the efficiency of scene embedding.
		cfg_path: str, the path for yaml file including parameters value
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
		embeddings_list = [model(batch[0]) for batch in loader]

	# Concatenate mini-batches back together and unflatten the frames
	# to reconstruct the audio batches
	embeddings = torch.cat(embeddings_list, dim=0)
	embeddings = embeddings.unflatten(0, (audio_batches, num_frames))

	return embeddings, timestamps


def get_scene_embeddings(
	audio_list: List,
	model: torch.nn.Module,
	cfg_path: str = 'hear/config.yaml'
) -> Tensor:
	"""
	This function returns a single embedding for each audio clip. 
	Args:
		audio_list: list of torch tensor audios (audios should be resampled to 16kHz).
		model: Loaded model.
		cfg_path: 
	Returns:
		- embeddings, A float32 Tensor with shape
			(n_sounds, model.scene_embedding_size).
	"""
	# Check if device has cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
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
						).to(device)
	mean, std = utils.compute_scene_stats(audio_list, to_melspec)
	# Calculate embeddings
	embeddings = []
	with torch.no_grad():
		for audio in tqdm(audio_list, desc=f'Generating Embeddings...', total=len(audio_list)):
			lms = ((to_melspec(audio.to(device).unsqueeze(0)) + torch.finfo(torch.float).eps).log()).unsqueeze(0)
			lms = (lms - mean) / std
			embedding = model(lms)
			embeddings.append(embedding)
	embeddings = torch.cat(embeddings, dim=0)
	return embeddings