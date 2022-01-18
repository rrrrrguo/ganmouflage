"""Base file for starting training
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import matplotlib
from data.utils import collate_fn
import yaml
from functools import partial
from warmup_scheduler import GradualWarmupScheduler
from data.render_depth import DepthRenderer
from trainer.training import Trainer
from model_v4.build_models import get_models
from data.dataset import get_dataset_single_scene
import random, os
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml
torch.backends.cudnn.benchmark = False
matplotlib.use('Agg')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def worker_init_fn(id,BASE_SEED):
    np.random.seed(BASE_SEED+id)

BASE_SEED=1234


def get_optimizers(models, cfg):
    model_g = models['generator']
    model_d = models['discriminator']

    lr_g = cfg['training']['lr_g']
    lr_d = cfg['training']['lr_d']
    betas=cfg['training']['betas']

    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=lr_g,betas=betas)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=lr_d,betas=betas)


    optimizers = {
        'generator': optimizer_g,
        'discriminator': optimizer_d,
    }
    return optimizers


def get_schedulers(optimizers,cfg):
    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(
        optimizers['generator'],milestones=cfg['training']['lr_step'],
        gamma=cfg['training']['gamma']
    )
    scheduler_g = GradualWarmupScheduler(
        optimizers['generator'], 1.0,
        cfg['training']['lr_warmup'], scheduler_g
    )

    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        optimizers['discriminator'],milestones=cfg['training']['lr_step'],
            gamma=cfg['training']['gamma']
    )
    scheduler_d = GradualWarmupScheduler(
        optimizers['discriminator'], 1.0,
        cfg['training']['lr_warmup'], scheduler_d
    )

    schedulers={
        'generator':scheduler_g,
        'discriminator':scheduler_d,
    }
    return schedulers


def get_renderer(cfg, device='cuda'):
    target_size = cfg['data']['scene']['target_size']
    renderer = DepthRenderer(target_size, device=device)
    return renderer


def get_trainer(models, optimizers, schedulers, renderer, cfg, device='cuda'):
    out_dir = cfg['training']['out_dir']

    print_every = cfg['training']['print_every']
    visualize_every = cfg['training']['visualize_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    backup_every = cfg['training']['backup_every']

    model_selection_metric = cfg['training']['model_selection_metric']
    model_selection_mode = cfg['training']['model_selection_mode']

    multi_gpu = cfg['training']['multi_gpu']

    trainer = Trainer(
        models['generator'], models['discriminator'],
        optimizers['generator'], optimizers['discriminator'],
        schedulers['generator'], schedulers['discriminator'],
        renderer,

        w_pix_ren=cfg['training']['w_pix_ren'],
        w_pix_ref=cfg['training']['w_pix_ref'],
        w_gan_ren=cfg['training']['w_gan_ren'],
        w_gan_ref=cfg['training']['w_gan_ref'],

        pix_ren_type=cfg['training']['pix_ren_type'],
        pix_ref_type=cfg['training']['pix_ref_type'],
        gan_type=cfg['training']['gan_type'],
        lambda_gp=cfg['training']['lambda_gp'],

        max_grad_norm=cfg['training']['max_grad_norm'],
        crop_size=cfg['training']['crop_size'],
        n_render_train=cfg['data']['scene']['n_views_sup'],
        n_render_val=cfg['data']['scene']['n_views_val'],
        n_ref=cfg['data']['scene']['n_views_ref'],
        transform_crop=cfg['training']['crop_transform'],
        update_g_every=cfg['training']['update_g_every'],

        multi_gpu=multi_gpu,
        out_dir=out_dir,
        model_selection_metric=model_selection_metric,
        model_selection_mode=model_selection_mode,
        print_every=print_every,
        visualize_every=visualize_every,
        checkpoint_every=checkpoint_every,
        backup_every=backup_every,
        validate_every=validate_every,
        device=device,
    )
    return trainer

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Train a Gamouflague.'
    )
    parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='Local process rank.')
    parser.add_argument('--conf_file', type=str, help='Path to config file.')
    parser.add_argument('--scene',type=str,default='scene1',
                        help="Scene name to use, will overwrite config")
    parser.add_argument('--log_dir',type=str,default='result/test_run',help="Path to output directory")
    parser.add_argument('--animals',action='store_true',help="Whether to train on animal shapes")
    args = parser.parse_args()
    seed_everything(BASE_SEED+args.local_rank)
    args.is_master = args.local_rank == 0
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device(f"cuda:{args.local_rank}")
    print(args.device)

    dist.init_process_group(backend='nccl', init_method='env://')
    with open(args.conf_file, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['data']['scene']['scene_name']=args.scene
    cfg['training']['out_dir']=args.log_dir
    cfg['data']['fake_cube']=not args.animals
    if args.animals:
        if args.scene in ['bookshelf-real','mit-37','walden-tree1']:
            cfg['data']['scene']['cube_scale']=1.5
        else:
            cfg['data']['scene']['cube_scale']=1.8
    else:
        if args.scene in ['bookshelf-real','walden-tree1']:
            cfg['data']['scene']['cube_scale']=0.8
        else:
            cfg['data']['scene']['cube_scale']=1
    if args.is_master:
        print(yaml.dump(cfg))

    models = get_models(cfg, device=args.device)
    models['generator']=torch.nn.SyncBatchNorm.convert_sync_batchnorm(models['generator'])
    models['generator']=DDP(
        models['generator'],        
        device_ids=[args.local_rank],
        output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True
    )
    models['discriminator']=DDP(
        models['discriminator'],        
        device_ids=[args.local_rank],
        output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True
    )

    optimizers = get_optimizers(models, cfg)
    schedulers=get_schedulers(optimizers,cfg)
    renderer = get_renderer(cfg, device=args.device)


    datasets = get_dataset_single_scene(
        cfg, debug=False,
        scene_name=cfg['data']['scene']['scene_name'],
        fake_cube=cfg['data']['fake_cube'],)

    train_loader = DataLoader(
        datasets['train'], 
        batch_size=cfg['training']['batch_size'],
        sampler=DistributedSampler(datasets['train'],shuffle=True),
        collate_fn=collate_fn, 
        num_workers=cfg['training']['num_workers'],
        worker_init_fn=partial(worker_init_fn,BASE_SEED=BASE_SEED+100*args.local_rank),
        drop_last=True)
    val_loader = DataLoader(
        datasets['valid'], 
        batch_size=cfg['training']['batch_size'], 
        sampler=DistributedSampler(datasets['valid'],shuffle=False),
        collate_fn=collate_fn, 
        num_workers=cfg['training']['num_workers'],
        worker_init_fn=partial(worker_init_fn,BASE_SEED=BASE_SEED+100*args.local_rank))
    trainer = get_trainer(models, optimizers,schedulers, renderer, cfg, device=args.device)
    with open(cfg['training']['out_dir'] + '/config.yaml',"w+") as f:
        yaml.dump(cfg,f)
    epoch=cfg['training']['max_epoch']
    torch.cuda.empty_cache()
    trainer.train(
        train_loader, 
        val_loader, None,
        exit_after=-1, n_epochs=epoch)
