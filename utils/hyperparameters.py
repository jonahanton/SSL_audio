import argparse 

MODELS = [
	'resnet50', 'resnet50_ReGP_NRF',
	'resnet18', 'resnet18_ReGP_NRF',
	'audiontt',
	'vit_base', 'vit_small', 'vit_tiny',
	'vitc_base', 'vitc_small', 'vitc_tiny'
]

DATASETS = [
	'fsd50k',
	'audioset',
	'librispeech',
	'fsd50k+librispeech',
	'cifar10'
]

OPTIMIZERS = [
	'Adam', 
	'AdamW', 
	'SGD', 
	'LARS',
]

def get_hyperparameters():
	parser = [get_std_parameters()]
	return parser


def get_std_parameters():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--model_type', default='audiontt', type=str, choices=MODELS)
	parser.add_argument('--dataset', default='fsd50k', type=str, choices=DATASETS)
	parser.add_argument('--epochs', default=100, type=int)
	parser.add_argument('--lr_schedule', action='store_true', default=False)
	parser.add_argument('--epoch_save_f', default=20, type=int)
	parser.add_argument('--epoch_eval_f', default=5, type=int)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--lmbda', type=float, default=0.005)
	parser.add_argument('--alpha', type=float, default=1)
	parser.add_argument('--projector_out_dim', default=256, type=int)
	parser.add_argument('--projector_n_hidden_layers', default=1, type=int)
	parser.add_argument('--projector_hidden_dim', default=8192, type=int)
	parser.add_argument('--local_crops_number', type=int, default=0)
	parser.add_argument('--local_crops_size', nargs='+', type=int, default=[16, 16])
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
	parser.add_argument('--use_fp16_eval', action='store_true', default=False)
	parser.add_argument('--name', type=str, default='')
	parser.add_argument('--squeeze_excitation', action='store_true', default=False)
	parser.add_argument('--mask', action='store_true', default=False)
	parser.add_argument('--mask_ratio', type=float, default=0)
	parser.add_argument('--random_mask_ratio', action='store_true', default=False)
	parser.add_argument('--mask_ratio_schedule', action='store_true', default=False)
	parser.add_argument('--mask_beta', type=float, default=0.5)
	parser.add_argument('--use_learned_pos_embd', action='store_true', default=False)
	parser.add_argument('--use_cls', action='store_true', default=True)
	parser.add_argument('--use_mean_pool', action='store_true', default=False)
	parser.add_argument('--patch_size', nargs='+', type=int, default=[16, 16])
	parser.add_argument('--masked_recon', action='store_true', default=False)
	parser.add_argument('--stop_gradient', action='store_true', default=False)
	parser.add_argument('--predictor', action='store_true', default=False)
	parser.add_argument('--save_base_dir', type=str, default='')

	parser.add_argument('--optimizer', type=str, default=None)
	parser.add_argument('--lr', type=float, default=None)
	parser.add_argument('--lr_weights', type=float, default=None)
	parser.add_argument('--lr_biases', type=float, default=None)
	parser.add_argument('--wd', type=float, default=None)
	return parser 


def setup_hyperparameters(args):
	if 'vit' in args.model_type:
		args.optimizer = 'AdamW' if args.optimizer is None else args.optimizer
		args.lr = 1e-4 * args.batch_size / 128 if args.lr is None else args.lr
		args.wd = 0.06 if args.wd is None else args.wd
	else:
		args.optimizer = 'LARS' if args.optimizer is None else args.optimizer
		args.lr_weights = 0.4 * args.batch_size / 128 if args.lr_weights is None else args.lr_weights
		args.lr_biases = 0.0048 * args.batch_size / 128 if args.lr_biases is None else args.lr_biases
		args.wd = 1e-5 if args.wd is None else args.wd
