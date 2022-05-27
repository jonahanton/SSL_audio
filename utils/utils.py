import torch

import numpy as np

import logging
import sys
from pathlib import Path
from easydict import EasyDict
import yaml


def load_yaml_config(path_to_config):
    """
    Loads yaml configuration settings as an EasyDict object.
    Taken from https://github.com/nttcslab/byol-a/
    """
    path_to_config = Path(path_to_config)
    assert path_to_config.is_file()
    with open(path_to_config) as f:
        yaml_contents = yaml.safe_load(f)
    cfg = EasyDict(yaml_contents)
    return cfg


def get_std_logging():
    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s',
        level=logging.INFO,
        )
    return logging


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """Taken from https://github.com/facebookresearch/dino/"""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))

    return schedule 


def get_params_groups(model):
    """Taken from https://github.com/facebookresearch/dino/"""
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]