import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import average_precision_score, roc_auc_score
import wandb
import argparse
import pandas as pd
from tqdm import tqdm

import utils
import datasets
from model import ResNet, ViT, BYOLAv2encoder


class Net(nn.Module):
	def __init__(self, cfg, num_class, pretrained_path, dataset):
		super(Net, self).__init__()
		self.cfg = cfg

		# encoder
		if cfg.model_type == 'resnet':
			self.f = ResNet(dataset=dataset).f
			out_dim = 2048
		elif cfg.model_type == 'vit_base':
			self.f = ViT(dataset=dataset, size='base', latent=cfg.latent).f
			out_dim = self.f.embed_dim 
		elif cfg.model_type == 'byola':
			self.f = BYOLAv2encoder(dataset=dataset).f
			out_dim = self.f.d
	
		# classifier
		self.fc = nn.Linear(out_dim, num_class, bias=True)
		
		# load weights
		self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

	def forward(self, x):
		x = self.f(x)
		if self.cfg.model_type == 'vit_base':
			x = x[:, 0].contiguous()
		feature = torch.flatten(x, start_dim=1)
		out = self.fc(feature)
		return out


def train_val(args, net, data_loader, train_optimizer, wandb_run):
	is_train = train_optimizer is not None
	net.train() if is_train else net.eval()

	total_loss, total_num, data_bar = 0.0, 0, tqdm(data_loader)
	if args.dataset == 'fsd50k':
		all_targets, all_preds = [], []
	else:
		total_correct_1, total_correct_5 = 0.0, 0.0 
	with (torch.enable_grad() if is_train else torch.no_grad()):
		for data, target in data_bar:
			data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
			out = net(data)
			loss = loss_criterion(out, target)

			if is_train:
				train_optimizer.zero_grad()
				loss.backward()
				train_optimizer.step()

			total_num += data.size(0)
			total_loss += loss.item() * data.size(0)
			if args.dataset == 'fsd50k':
				all_targets.append(target)
				all_preds.append(out.sigmoid())
				data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} model: {}'
										.format('Train' if is_train else 'Test', epoch, args.epochs, total_loss / total_num,
												args.model_path.split('/')[-1]))
				if is_train:
					wandb_run.log({'Loss': total_loss / total_num})
			else:
				prediction = torch.argsort(out, dim=-1, descending=True)
				total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
				total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

				data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}% model: {}'
										.format('Train' if is_train else 'Test', epoch, args.epochs, total_loss / total_num,
												total_correct_1 / total_num * 100, total_correct_5 / total_num * 100,
												args.model_path.split('/')[-1]))
				if is_train:
					wandb_run.log({
						'Loss': total_loss / total_num,
						'ACC@1': total_correct_1 / total_num * 100,
						'ACC@5': total_correct_5 / total_num * 100,
					})

	if args.dataset == 'fsd50k':
		all_preds = torch.cat(all_preds, dim=0)
		all_targets = torch.cat(all_targets, dim=0)
		mAP = average_precision_score(y_true=all_targets.detach().cpu(), y_score=all_preds.detach().cpu(), average='macro') 
		AUC = roc_auc_score(y_true=all_targets.detach().cpu(), y_score=all_preds.detach().cpu(), average='macro') 
		return total_loss / total_num, {'mAP': mAP, 'AUC': AUC}
	else:
		return total_loss / total_num, {'acc_1': total_correct_1 / total_num * 100, 'acc_5': total_correct_5 / total_num * 100}


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Linear Evaluation')
	parser.add_argument('--dataset', default='fsd50k', type=str, help='Dataset: cifar10 or tiny_imagenet or stl10 or fsd50k or nsynth')
	parser.add_argument('--num_classes', default=200, type=int)
	parser.add_argument('--model_path', type=str, default=None, help='The base string of the pretrained model path')
	parser.add_argument('--model_type', default='resnet', type=str, help='Encoder: resnet or vit (base)')
	parser.add_argument('--batch_size', type=int, default=256, help='Number of samples in each mini-batch')
	parser.add_argument('--epochs', type=int, default=50, help='Number of sweeps over the dataset to train')
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--wd', type=float, default=1e-6)
	# for audio processing
	parser.add_argument('--unit_sec', type=float, default=0.95)
	parser.add_argument('--crop_frames', type=int, default=96)
	parser.add_argument('--sample_rate', type=int, default=16000)
	parser.add_argument('--n_fft', type=int, default=1024)
	parser.add_argument('--win_length', type=int, default=1024)
	parser.add_argument('--hop_length', type=int, default=160)
	parser.add_argument('--n_mels', type=int, default=64)
	parser.add_argument('--f_min', type=int, default=60)
	parser.add_argument('--f_max', type=int, default=7800)
	# load pre-computed lms 
	parser.add_argument('--load_lms', action='store_true', default=False)
	# end-to-end finetune 
	parser.add_argument('--finetune', action='store_true', default=False)
	parser.add_argument('--num_workers', type=int, default=4)

	args = parser.parse_args()

	# wandb init
	wandb_run = wandb.init(
			project='barlow twins {} linear'.format(args.dataset),
			config=args,
			settings=wandb.Settings(start_method="fork"),
		)

	if args.dataset == 'cifar10':
		train_data = torchvision.datasetsCIFAR10(root='data', train=True,\
			transform=utils.CifarPairTransform(train_transform=True, pair_transform=False), download=True)
		test_data = torchvision.datasetsCIFAR10(root='data', train=False,\
			transform=utils.CifarPairTransform(train_transform=False, pair_transform=False), download=True)
	elif args.dataset == 'fsd50k':
		# fsd50k [mean, std] (lms)
		norm_stats = [-4.950, 5.855]
		train_data = datasets.FSD50K(args, train=True, transform=utils.AudioPairTransform(train_transform=True, pair_transform=False),
									 norm_stats=norm_stats)
		test_data = datasets.FSD50K(args, train=False, transform=utils.AudioPairTransform(train_transform=False, pair_transform=False), 
									norm_stats=norm_stats)
	elif args.dataset == 'nsynth':
		# nysnth [mean, std] (lms)
		norm_stats = [-8.82, 7.03]
		train_data = datasets.NSynth(args, split='train', transform=utils.AudioPairTransform(train_transform=True, pair_transform=False),
									 norm_stats=norm_stats)
		test_data = datasets.NSynth(args, split='test', transform=utils.AudioPairTransform(train_transform=False, pair_transform=False), 
									norm_stats=norm_stats)

	train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	
	model = Net(args, num_class=args.num_classes, pretrained_path=args.model_path, dataset=args.dataset).cuda()
	if not args.finetune:
		for param in model.f.parameters():
			param.requires_grad = False

	if args.finetune and 'vit' in args.model_type:
		optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.lr) 
	else:
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

	if args.dataset == 'fsd50k':
		loss_criterion = nn.BCEWithLogitsLoss()
		results = {'train_loss': [], 'train_mAP': [], 'train_AUC': [],
				   'test_loss': [], 'test_mAP': [], 'test_AUC': []}
	else:
		loss_criterion = nn.CrossEntropyLoss()
		results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
				   'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

	save_name = args.model_path.split('.pth')[0] + '_linear_{}.csv'.format(args.dataset)

	best_acc = 0.0
	for epoch in range(1, args.epochs + 1):
		train_loss, train_stats = train_val(args, model, train_loader, optimizer, wandb_run)
		results['train_loss'].append(train_loss)
		if args.dataset == 'fsd50k':
			results['train_mAP'].append(train_stats['mAP'])
			results['train_AUC'].append(train_stats['AUC'])
			wandb_run.log({
				'train_mAP': train_stats['mAP'],
				'train_AUC': train_stats['AUC'],
			})
		else:
			results['train_acc@1'].append(train_stats['acc_1'])
			results['train_acc@5'].append(train_stats['acc_5'])
			wandb_run.log({
				'train_acc@1': train_stats['acc_1'],
				'train_acc@5': train_stats['acc_5'],
			})
		test_loss, test_stats = train_val(model, test_loader, None, wandb_run)
		results['test_loss'].append(test_loss)
		if args.dataset == 'fsd50k':
			results['test_mAP'].append(test_stats['mAP'])
			results['test_AUC'].append(test_stats['AUC'])
			wandb_run.log({
				'test_mAP': test_stats['mAP'],
				'test_AUC': test_stats['AUC'],
			})
		else:
			results['test_acc@1'].append(test_stats['acc_1'])
			results['test_acc@5'].append(test_stats['acc_5'])
			wandb_run.log({
				'test_acc@1': test_stats['acc_1'],
				'test_acc@5': test_stats['acc_5'],
			})
		# save statistics
		data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
		data_frame.to_csv(save_name, index_label='epoch')