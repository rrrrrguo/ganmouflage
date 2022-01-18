import torch
import torch.nn as nn
from torchvision.models import vgg19, vgg16
from torchvision.transforms import Normalize
import torch.nn.functional as F
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
VGG16_Layer = [3, 8, 15, 22]
VGG19_Layer = [3, 8, 17, 26]


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization = Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        self.vgg = vgg16(pretrained=True).features
        self.use_layer = VGG16_Layer
    def forward(self, x, y):
        y = self.normalization(y)
        x = self.normalization(x)
        loss=0
        for i in range(max(self.use_layer) + 1):
            x = self.vgg[i](x)
            y = self.vgg[i](y)
            if i in self.use_layer:
                loss+=F.l1_loss(x,y)
        return loss
