"""
Utility function to calculate the spectrogram mean and std of AudioSet-2M.
"""
import torch

import argparse
import sys
sys.path.append('../../../')
import os
import numpy as np
from tqdm import tqdm 
from pprint import pprint

from data_manager.audioset import AudioSet
from utils import utils



def get_args_parser():
    
    parser = argparse.ArgumentParser(description='Calculate dataset normalization stats', add_help=False)
    parser.add_argument('--config-path', type=str, default='./configs/pretrain/config.yaml',
                        help='path to .yaml config file')
    parser.add_argument('--n-norm-calc', type=int, default=10000)
    return parser


def calculate_norm_stats(cfg, n_norm_calc=10000):

    # load dataset
    dataset = AudioSet(cfg, n_views=1)

    # calculate norm stats (randomly sample n_norm_calc points from dataset)
    idxs = np.random.randint(0, len(dataset), size=n_norm_calc)
    lms_vectors = []
    for i in tqdm(idxs):
        lms_vectors.append(dataset[i][0])
    lms_vectors = torch.stack(lms_vectors)
    norm_stats = lms_vectors.mean(), lms_vectors.std() + torch.finfo().eps

    print(f'Dataset contains {len(dataset)} files with normalizing stats\n'
          f'mean: {norm_stats[0]}\t std: {norm_stats[1]}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Norm-stats', parents=[get_args_parser()])
    args = parser.parse_args()

    # load training params from .yaml config file
    cfg = utils.load_yaml_config(args.config_path)
    # update config with any remaining arguments from args
    utils.update_cfg_from_args(cfg, args)
    
    # calculate norm stats
    calculate_norm_stats(cfg, args.n_norm_calc)