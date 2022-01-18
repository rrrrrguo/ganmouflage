# Modified from https://github.com/usuyama/pytorch-unet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNet34(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()

        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        #self.layer0_1x1 = convrelu(64, 64, 1, 0)
        # size=(N, 64, x.H/4, x.W/4)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.normalize = normalize

    def forward(self, input):
        #[b,c,h,w]
        h,w=input.shape[2:]
        if self.normalize:
            input = normalize_imagenet(input)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        #layer4 = self.layer4(layer3)

        features= [layer0,layer1,layer2, layer3]
        for i in range(len(features)):
            features[i]=F.interpolate(
                features[i],
                size=(h,w),
                mode='bilinear',
                align_corners=True
            )
            #print(features[i].shape)
        return features


class ResNet34UNet(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()

        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        #self.layer0_1x1 = convrelu(64, 64, 1, 0)
        # size=(N, 64, x.H/4, x.W/4)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 256, 3, 1)
        self.conv_up2 = convrelu(128 + 256, 128, 3, 1)
        self.conv_up1 = convrelu(64 + 128, 128, 3, 1)
        #self.conv_up0 = convrelu(64 + 128, 128, 3, 1)

        #self.conv_original_size0 = convrelu(3, 64, 3, 1)
        #self.conv_original_size1 = convrelu(64, 64, 3, 1)
        #self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        self.normalize = normalize

    def forward(self, input):
        if self.normalize:
            input = normalize_imagenet(input)

        #x_original = self.conv_original_size0(input)
        #x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)  # 32
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)  # 16
        features_16 = x
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)  # 8
        features_8 = x
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)  # 4
        features_4 = x

        #x = self.upsample(x)
        #layer0 = self.layer0_1x1(layer0)
        #x = torch.cat([x, layer0], dim=1)
        #x = self.conv_up0(x)
        #feature_2 = x
        #x = self.upsample(x)
        #x = torch.cat([x, x_original], dim=1)
        #x = self.conv_original_size2(x)
        #feature_1 = x

        return [features_4, features_8, features_16]


class ResNet18UNetHighRes(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        # size=(N, 64, x.H/4, x.W/4)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 256, 1, 0)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 256, 128, 3, 1)
        self.conv_up2 = convrelu(128 + 128, 128, 3, 1)
        self.conv_up1 = convrelu(64 + 128, 64, 3, 1)
        self.conv_up0 = convrelu(64 + 64, 64, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 64, 32, 3, 1)
        self.normalize = normalize

    def forward(self, input):
        if self.normalize:
            input = normalize_imagenet(input)

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)  # 32
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)  # 16
        features_16 = x
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)  # 8
        #features_8 = x
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)  # 4
        features_4 = x

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        #feature_2 = x
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        feature_1 = x

        return [feature_1, features_4, features_16]


def normalize_imagenet(x):
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


def bilinear_interpolation(feature_map, coords,mode='bilinear'):
    '''
    Args:
        feature_map: [B,C,H,W] of image feature map
        coords: [B,K,2] of normalized coordinates, range(-1,1)

    Returns:
        features [B,C,K]
    '''
    features = F.grid_sample(feature_map, coords.unsqueeze(
        2),mode=mode, padding_mode='zeros', align_corners=True).squeeze(-1)  # grid [B,K,1,2], features [B,C,K,1]
    return features  # [B,C,K]


def bilinear_interpolation_list(feature_maps, coords,mode='bilinear'):
    features = [bilinear_interpolation(fm, coords,mode) for fm in feature_maps]
    features = torch.cat(features, dim=1)
    return features
