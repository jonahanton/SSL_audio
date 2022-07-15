import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

from vision_transformer import mae


class ViT(nn.Module):
    def __init__(self, feature_dim=128, dataset='fsd50k', size='base', latent='cls'):
        super(ViT, self).__init__()
        self.latent = latent 

        # encoder
        if size == 'base':
            self.f = mae.mae_vit_base_patch16x16()
        embed_dim = self.f.embed_dim
        bottleneck_dim = int(embed_dim / 4)
        # projection head
        self.g = nn.Sequential(nn.Linear(embed_dim, bottleneck_dim, bias=False), nn.BatchNorm1d(bottleneck_dim),
                               nn.ReLU(inplace=True), nn.Linear(bottleneck_dim, feature_dim, bias=True))

    def forward(self, x, mask_ratio=0.):
        x = self.f(x, mask_ratio=mask_ratio)
        if self.latent == 'cls':
            x = x[:, 0]
        elif self.latent == 'pool':
            x = torch.mean(x[:, 1:], dim=1)
        feature = x.contiguous()
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class ResNet(nn.Module):
    def __init__(self, feature_dim=128, dataset='cifar10'):
        super(ResNet, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                if dataset == 'fsd50k':
                    module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
                else:
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if dataset == 'cifar10':
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            elif dataset == 'tiny_imagenet' or dataset == 'stl10' or dataset == 'fsd50k':
                if not isinstance(module, nn.Linear):
                    self.f.append(module)

        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x, mask_ratio=0.):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)



if __name__ == "__main__":

    model = ViT(dataset='fsd50k')
    x = torch.randn(3, 1, 64, 96)
    feature, out = model(x)
    print(feature.shape)
    print(out.shape)