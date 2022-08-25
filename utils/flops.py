import argparse
import os
import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from models import mae, resnet
from model import AudioNTT2022

MODELS = [
	'resnet18', 'resnet18_ReGP_NRF',
	'audiontt',
	'vit_base', 'vit_small', 'vit_tiny',
	'vitc_base', 'vitc_small', 'vitc_tiny'
]


def obtain_model_stats(model, mask_ratio=0, out_path=None):
	kwargs = {}
	if mask_ratio > 0:
		kwargs['mask_ratio'] = mask_ratio
	with torch.cuda.device(0):
		batch_size = 2
		flops, macs, params = get_model_profile(model=model, # model
										input_shape=(batch_size, 1, 64, 96), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
										# args=None, # list of positional arguments to the model.
										kwargs=kwargs, # dictionary of keyword arguments to the model.
										print_profile=True, # prints the model graph with the measured profile attached to each module
										detailed=False, # print the detailed profile
										module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
										top_modules=1, # the number of top modules to print aggregated profile
										warm_up=5, # the number of warm-ups before measuring the time of each module
										as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
										output_file=out_path, # path to the output file. If None, the profiler prints to stdout.
										ignore_modules=None) # the list of modules to ignore in the profiling


def run(model_type='audiontt', mask_ratio=0, patch_size=(16, 16), outpath=None):

	if model_type == 'resnet18':
		model = resnet.resnet18()
		model.fc = nn.Identity()
	elif model_type == 'resnet18_ReGP_NRF':
		model = resnet.resnet18_ReGP_NRF()
		model.fc = nn.Identity()
	elif model_type == 'audiontt':
		model = AudioNTT2022()
	elif 'vit' in model_type:
		size = model_type.split('_')[-1]
		conv_stem_bool = model_type.split('_')[0] == 'vitc'
		model = mae.get_mae_vit(size, patch_size, conv_stem_bool)
	
	obtain_model_stats(model, mask_ratio, outpath)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--model_type', default='audiontt', nargs='+', type=str, choices=MODELS)
	parser.add_argument('--mask_ratio', type=float, default=0)
	parser.add_argument('--patch_size', nargs='+', type=int, default=[16, 16])
	args = parser.parse_args()

	for model_type in args.model_type:
		
		save_name = model_type
		if 'vit' in model_type:
			save_name += f'_mask_ratio={args.mask_ratio}_patch_size={args.patch_size[0]}x{args.patch_size[1]}'
		log_dir = f'logs/flops/{model_type}'
		os.makedirs(log_dir, exist_ok=True)
		out_path = os.path.join(log_dir, f'{save_name}.log')

		run(model_type, args.mask_ratio, args.patch_size, out_path)