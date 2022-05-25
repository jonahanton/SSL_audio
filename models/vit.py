"""
A PyTorch implementation of Vision Transformers [Dosovitskiy et al., 2020], adapted for our purposes.
Mostly copy-paste from timm library:
	https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
Other references:
	https://github.com/facebookresearch/mae/blob/main/models_mae.py
	https://github.com/YuanGongND/ssast/blob/main/src/models/ast_models.py#L76
	https://github.com/facebookresearch/msn/blob/main/src/deit.py
"""
import math
from functools import partial
import numpy as np

import torch 
import torch.nn as nn


class VisionTransformer(nn.Module):
	def __init__(self):
		pass
