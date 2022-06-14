"""
References:
    https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/eval_knn.py
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score

from utils import utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # for sklearn UserWarning 


@torch.no_grad()
def mlknn_classifier(train_features, train_labels, test_features, test_labels, k=200, T=0.07, num_classes=527):
    
    if isinstance(train_features, torch.Tensor):
        train_features = train_features.cpu().numpy()
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.cpu().numpy()
    if isinstance(test_features, torch.Tensor):
        test_features = test_features.cpu().numpy()
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.cpu().numpy()
    
    print('Calculating knn mAP')
    classifier = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights=lambda x: np.exp(x/T))
    classifier.fit(X=train_features, y=train_labels)
    pred_test_labels = classifier.predict(test_features)

    mAP = average_precision_score(y_true=test_labels, y_score=pred_test_labels, average='macro')
    return mAP


@torch.no_grad()
def extract_features(cfg, model, data_loader, use_cuda=False, multiscale=False):

    print('Extracting features')
    features = None
    for (y, labels, index) in data_loader:
        y = y.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        feats, _, _ = model(y, mask_ratio=0.)
        if cfg.model.encoder.latent == 'cls':
            # return cls token as global clip representation
            feats = feats[:, 0]
        else:
            # return mean pool over patch embeddings as global clip representation
            feats = torch.mean(feats[:, 1:], dim=1)

        # init storage feature matrix
        if utils.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            labs = torch.zeros(len(data_loader.dataset), labels.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
                labs = labs.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")
            print(f"Storing labels into tensor of shape {labs.shape}")

        """get indexes from all processes"""
        y_all = torch.empty(utils.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = dist.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        """share features between processes"""
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = dist.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        """share labels between processes"""
        labels_all = torch.empty(
            utils.get_world_size(),
            labels.size(0), 
            labels.size(1),
            dtype=labels.dtype,
            device=labels.device,
        )
        labels_l = list(labels_all.unbind(0))
        labels_all_reduce = dist.all_gather(labels_l, labels, async_op=True)
        labels_all_reduce.wait()

        # update storage feature matrix
        if utils.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l).detach())
                labs.index_copy_(0, index_all, torch.cat(labels_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).detach().cpu())
                labs.index_copy_(0, index_all.cpu(), torch.cat(labels_l).cpu())

    return features, labs