"""
Wrapper for our pre-trained MST models for the HEAR 2021 Challenge.

References:
    https://hearbenchmark.com/hear-api.html
    https://github.com/nttcslab/msm-mae/blob/main/hear/hear_msm/sample.py
"""

import sys
sys.path.append('../..')

import torch
import torch.nn as nn

from models.runtime import RuntimeMST


def load_model(model_file_path='/rds/general/user/jla21/ephemeral/SSL_audio/checkpoint/example_model.pth.tar'):
    model = RuntimeMST(weight_file=model_file_path)
    if torch.cuda.is_available():
        model.cuda()
    return model


def get_scene_embeddings(audio, model):
    model.eval()
    with torch.no_grad():
        return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    model.eval()
    with torch.no_grad():
        return model.get_timestamp_embeddings(audio)

