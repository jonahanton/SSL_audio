import torch
import torch.nn as nn
import torchaudio

import random 

from audiomentations import Compose, TimeStretch, PitchShift
from data_manager.augmentations import Mixup, MixGaussianNoise


def make_transforms_pretrain(cfg):
	
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
	

def make_transforms_eval(cfg):
	pass