from audiomentations import Compose, TimeStretch, PitchShift
from data_manager.augmentations_lms import Mixup, MixGaussianNoise


def make_transforms(cfg):
	
	# transforms to raw waveform (.wav) -> time strech, pitch shift
	wav_transform = Compose([
		TimeStretch(min_rate=cfg.transform_min_rate, max_rate=cfg.transform_max_rate, p=cfg.transform_ts_p)
		PitchShift(min_semitones=cfg.transform_min_semitones, max_semitones=cfg.transform_max_semitones, p=cfg.transform_ps_p),
	])
	
	# transforms to lms -> mixup (from BYOL-A), and/or addition of gaussian noise
	lms_transform = []
	if cfg.transform_mixup:
		lms_transform.append(Mixup(ratio=cfg.transform_mixup_ratio))
	if cfg.transform_gaussnoise:
		lms_transform.append(MixGaussianNoise(ratio=cfg.transform_gaussnoise_ratio))
	lms_transform = nn.Sequential(*lms_transform)
	
	return wav_transform, lms_transform
	
