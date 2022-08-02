import argparse 


DATASETS = [
	'fsd50k',
	'audioset',
	'librispeech',
	'fsd50k+librispeech',
]

OPTIMIZERS = [
	'Adam', 
	'AdamW', 
	'SGD', 
	'LARS',
]

def get_hyperparameters(args):

	model_type = args.model_type
	parsers = [get_std_parameters()]
	if 'vit' in model_type:
		parsers.append(get_vit_parameters())
	else:
		parsers.append(get_conv_parameters())
	return parsers


def get_std_parameters():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--dataset', default='fsd50k', type=str, choices=DATASETS)
	parser.add_argument('--epochs', default=100, type=int)
	parser.add_argument('--epoch_save_f', default=20, type=int)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--lmbda', type=float, default=0.005)
	parser.add_argument('--alpha', type=float, default=1)
	parser.add_argument('--projector_out_dim', default=256, type=int)
	parser.add_argument('--projector_n_hidden_layers', default=1, type=int)
	parser.add_argument('--projector_hidden_dim', default=8192, type=int)
	parser.add_argument('--unit_sec', type=float, default=0.95)
	parser.add_argument('--crop_frames', type=int, default=96)
	parser.add_argument('--sample_rate', type=int, default=16000)
	parser.add_argument('--n_fft', type=int, default=1024)
	parser.add_argument('--win_length', type=int, default=1024)
	parser.add_argument('--hop_length', type=int, default=160)
	parser.add_argument('--n_mels', type=int, default=64)
	parser.add_argument('--f_min', type=int, default=60)
	parser.add_argument('--f_max', type=int, default=7800)
	parser.add_argument('--num_workers', type=int, default=20)
	parser.add_argument('--mixup_ratio', type=float, default=0.2)
	parser.add_argument('--virtual_crop_scale', nargs='+', type=float, default=[1, 1.5])
	parser.add_argument('--HSIC', action='store_true', default=False)
	parser.add_argument('--mixup', action='store_true', default=True)
	parser.add_argument('--no_mixup', action='store_false', dest='mixup')
	parser.add_argument('--RRC', action='store_true', default=True)
	parser.add_argument('--no_RRC', action='store_false', dest='RRC')
	parser.add_argument('--RLF', action='store_true', default=True)
	parser.add_argument('--no_RLF', action='store_false', dest='RLF')
	parser.add_argument('--Gnoise', action='store_true', default=False)
	parser.add_argument('--pre_norm', action='store_true', default=False)
	parser.add_argument('--post_norm', action='store_true', default=False)
	parser.add_argument('--load_lms', action='store_true', default=True)
	parser.add_argument('--load_wav', action='store_false', dest='load_lms')
	parser.add_argument('--distributed', action='store_true', default=False)
	parser.add_argument('--use_fp16', action='store_true', default=False)
	parser.add_argument('--name', type=str, default='')
	return parser 


def get_vit_parameters():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--lr', type=float, default=2e-5)
	parser.add_argument('--wd', type=float, default=0.24)
	parser.add_argument('--optimizer', type=str, default='AdamW', choices=OPTIMIZERS)
	parser.add_argument('--mask', action='store_true', default=False)
	parser.add_argument('--mask_ratio', type=float, default=0)
	parser.add_argument('--int_layers', action='store_true', default=False)
	parser.add_argument('--int_layer_step', type=int, default=3)
	parser.add_argument('--use_learned_pos_embd', action='store_true', default=False)
	parser.add_argument('--use_max_pool', action='store_true', default=False)
	return parser


def get_conv_parameters():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--lr_weights', type=float, default=0.4)
	parser.add_argument('--lr_biases', type=float, default=0.0048)
	parser.add_argument('--wd', type=float, default=0.24)
	parser.add_argument('--optimizer', type=str, default='LARS', choices=OPTIMIZERS)
	parser.add_argument('--squeeze_excitation', action='store_true', default=False)
	return parser