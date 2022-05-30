import torch
import torch.nn as nn
import torchaudio

import random 

from audiomentations import Compose, TimeStretch, PitchShift
from data_manager.augmentations import Mixup, MixGaussianNoise


def make_transforms(cfg):
	
	# transforms to raw waveform (.wav) -> time strech, pitch shift
	wav_transform = Compose([
		TimeStretch(
			min_rate=cfg.data.transform.min_rate, 
			max_rate=cfg.data.transform.max_rate,
			p=cfg.data.transform.ts_p,
		),
		PitchShift(
			min_semitones=cfg.data.transform.min_semitones,
			max_semitones=cfg.data.transform.max_semitones,
			p=cfg.data.transform.ps_p,
		),
	])
	
	# transforms to lms -> mixup (from BYOL-A), and/or addition of gaussian noise
	lms_transform = []
	if cfg.data.transform.mixup:
		lms_transform.append(Mixup(ratio=cfg.data.transform.mixup_ratio))
	if cfg.data.transform.gaussnoise:
		lms_transform.append(MixGaussianNoise(ratio=cfg.data.transform.gaussnoise_ratio))
	lms_transform = nn.Sequential(*lms_transform)
	
	return wav_transform, lms_transform
	

if __name__ == "__main__":

	from utils import utils
	cfg = utils.load_yaml_config('config.yaml')
	wav_transform, lms_transform = make_transforms(cfg)

	unit_length = int(0.95 * 16000)
	wav, sr = torchaudio.load('data/audioset/samples/--1yd6dcNOQ.wav')

	# if audio has 2 channels, convert to mono
	if wav.shape[0] == 2:
		wav = torch.mean(wav, dim=0).unsqueeze(0)
	wav = wav[0]  # (1, length) -> (length,)

	# zero padding to both ends
	length_adj = unit_length - len(wav)
	if length_adj > 0:
		half_adj = length_adj // 2
		wav = F.pad(wav, (half_adj, length_adj - half_adj))

	# random crop unit length wave
	length_adj = len(wav) - unit_length
	start = random.randint(0, length_adj) if length_adj > 0 else 0
	wav = wav[start:start + unit_length]

	torchaudio.save('wav.wav', wav1.unsqueeze(0), sample_rate=16000)
	wav_tf1 = wav_transform(wav.numpy(), sample_rate=16000)
	wav_tf1 = torch.tensor(wav_tf1)
	wav_tf2 = wav_transform(wav.numpy(), sample_rate=16000)
	wav_tf2 = torch.tensor(wav_tf2)
	torchaudio.save('wav_tf1.wav', wav_tf1.unsqueeze(0), sample_rate=16000)
	torchaudio.save('wav_tf2.wav', wav_tf2.unsqueeze(0), sample_rate=16000)
