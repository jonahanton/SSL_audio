import torch 
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from models import resnet, mae
from model import ViT, AudioNTT2022

# resnet50
r50 = resnet.resnet50()
r50.fc = nn.Identity()
# resnet50 (ReGP+N.RF)
r50_ReGP_NRF = resnet.resnet50_ReGP_NRF()
r50_ReGP_NRF.fc = nn.Identity()
# audiontt
audiontt = AudioNTT2022()
# vit[tiny, small, base]
vit_tiny = ViT('tiny')
vit_small = ViT('small')
vit_base = ViT('base')

archs = {
    'resnet50':r50, 'resnet50_ReGP_NRF':r50_ReGP_NRF,
    'audiontt':audiontt,
    'vit_tiny':vit_tiny, 'vit_small':vit_small, 'vit_base':vit_base,
}

x = torch.randn((1, 1, 64, 96))
for k, v in archs.items():
    print(k)
    flops = FlopCountAnalysis(v, x)
    print(f"{flops.total():,}")