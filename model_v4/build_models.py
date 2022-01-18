from .modules.resnet_unet import ResNet18UNetHighRes,ResNet34UNet,ResNet34
from .modules.decoder import ImplicitDecoder
from .modules.texture_model import TextureNetwork
from .modules.patch import Discriminator


def get_models(cfg,device):
    discriminator = Discriminator(**cfg['model']['discriminator'])

    if cfg['model']['image_encoder']['type']=='resnet34unet':
        image_encoder=ResNet34UNet(**cfg['model']['image_encoder']['args'])
    elif cfg['model']['image_encoder']['type']=='resnet18unethigh':
        image_encoder=ResNet18UNetHighRes(**cfg['model']['image_encoder']['args'])
    else:
        image_encoder=ResNet34(**cfg['model']['image_encoder']['args'])

    decoder=ImplicitDecoder(**cfg['model']['decoder'])
    generator=TextureNetwork(image_encoder,decoder,**cfg['model']['args'])
    
    # Output dict
    models_out = {
        'generator': generator.to(device),
        'discriminator': discriminator.to(device)
    }
    return models_out