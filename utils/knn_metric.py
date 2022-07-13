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


def predict_knn(cfg, model, train_loader, test_loader):

    print('Extracting features...')
    train_features, train_labels = extract_features(cfg, model, train_loader)
    test_features, test_labels = extract_features(cfg, model, test_loader)
    
    print('Features are ready!\nStart the k-NN classification.')
    if utils.get_rank() == 0:
        
        knn_mAP = mlknn_classifier(
            train_features,
            train_labels,
            test_features,
            test_labels,
            cfg.knn.k,
            cfg.knn.T,
            cfg.knn.num_classes,
        )
        return knn_mAP


@torch.no_grad()
def extract_features(cfg, model, data_loader):
    model.eval()
    feature_bank, feature_labels = [], []

    for (inputs, labels) in data_loader:
        if cfg.model.encoder.type == 'transformer':
            feature, _, _ = model(inputs.cuda(non_blocking=True))
            if cfg.model.encoder.latent == 'cls':
                feature = feature[:, 0]
            else:
                feature = torch.mean(feature[:, 1:], dim=1)
        elif cfg.model.encoder.type.split('-')[0] == 'byola':
            feature = model(inputs.cuda(non_blocking=True), return_projection=False)
        else:
            feature = model(inputs.cuda(non_blocking=True))
        feature = feature.contiguous()
        feature = F.normalize(feature, dim=1)
        feature_bank.append(feature)
        feature_labels.append(labels.cuda(non_blocking=True))
    
    feature_bank = torch.cat(feature_bank, dim=0)
    feature_labels = torch.cat(feature_labels, dim=0)
    
    if cfg.meta.distributed:
        feature_bank_list = [torch.zeros_like(feature_bank) for _ in range(cfg.world_size)]
        dist.all_gather(feature_bank_list, feature_bank)
        feature_bank = torch.cat(feature_bank_list, dim=0)  # [N, D]

        feature_labels_list = [torch.zeros_like(feature_labels) for _ in range(cfg.world_size)]
        dist.all_gather(feature_labels_list, feature_labels)
        feature_labels = torch.cat(feature_labels_list, dim=0)

    return feature_bank, feature_labels


def mlknn_classifier(train_features, train_labels, test_features, test_labels, k=200, T=0.07, num_classes=527):
    
    if isinstance(train_features, torch.Tensor):
        train_features = train_features.cpu().numpy()
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.cpu().numpy()
    if isinstance(test_features, torch.Tensor):
        test_features = test_features.cpu().numpy()
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.cpu().numpy()
    
    classifier = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights=lambda x: np.exp(x/T))
    classifier.fit(X=train_features, y=train_labels)
    pred_test_labels = classifier.predict(test_features)

    mAP = average_precision_score(y_true=test_labels, y_score=pred_test_labels, average='macro')
    return mAP


