import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128, dataset='cifar10'):
        super(Model, self).__init__()

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

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)



if __name__ == "__main__":

    model = Model(dataset='fsd50k')
    print(model)
    x = torch.randn(1, 1, 64, 96)
    feature = model.f(x)
    print(feature.shape)