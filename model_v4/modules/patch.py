import torch
import torch.nn as nn
import functools

class Discriminator(nn.Module):
    def __init__(self,nc=3,nh=64,norm='batch'):
        super(Discriminator, self).__init__()
        if norm=='none':
            norm_fn=nn.Identity
        elif norm=='instance':
            norm_fn=nn.InstanceNorm2d
        else:
            norm_fn=nn.BatchNorm2d
        self.main = nn.Sequential(
            nn.Conv2d(nc, nh, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nh, nh * 2, 4, 2, 1, bias=False),
            norm_fn(nh * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nh * 2, nh * 4, 4, 2, 1, bias=False),
            norm_fn(nh * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nh * 4, nh * 8, 4, 2, 1, bias=False),
            norm_fn(nh * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nh * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)