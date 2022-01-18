import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from .basetrainer import BaseTrainer
from data.render_depth import toMatrix
import cv2
from data.utils import collate_fn
from model_v4.visualizer import TextureVisualizer
import matplotlib.pyplot as plt
import math
from .random_crop import get_bbox_from_mask, RandomResizedCropAroundTarget
from kornia.augmentation import RandomResizedCrop
from pytorch3d.structures import Meshes
from .perceptual_loss import PerceptualLoss

def to_device(data, device='cuda'):
    for k in data:
        #print(k,type(data[k]))
        if type(data[k]) is list:
            #print(k)
            data[k] = [x.to(device) for x in data[k]]
        else:
            data[k] = data[k].to(device)
    return data

def to_float(x):
    if type(x)==torch.Tensor:
        return x.item()
    else:
        return x

def prepare_input(data, device='cuda', cat_ref_mask=False,render_idx=(1,2),ref_idx=(0,1)):
    cam_K = data['cam_K'][:, render_idx[0]:render_idx[1]]
    n_views=cam_K.shape[1]
    obj2world = toMatrix(data['obj_rotation'], data['obj2world'])  # [B,4,4] normalize the positions, cube center is at origin
    world2camera = data['scene_Rts'][:, render_idx[0]:render_idx[1]]  # [B,nview,3,4]
    # transform matrix from obj space to camera space[B,3,4]
    obj2cams = torch.matmul(world2camera, obj2world.unsqueeze(1)) #[B,nview,3,4]
    background = data['scene_images'][:, render_idx[0]:render_idx[1]].clone() #[b,nviews,h,w,3]
    cube_diagonal = data['cube_diagonal'].unsqueeze(1).repeat(1,n_views) #[b,nviews]

    input_image = data['scene_images'][:, ref_idx[0]:ref_idx[1]].permute(0, 1,4, 2,3)  # [B,nview,3,h,w]
    mask = data['scene_masks'][:, ref_idx[0]:ref_idx[1]].unsqueeze(2)# [B,nview,1,h,w]
    if cat_ref_mask:
        input_image = torch.cat([input_image, mask], dim=2)
    cam_K_ref = data['cam_K'][:, ref_idx[0]:ref_idx[1]]
    cam_W_ref = torch.matmul(data['scene_Rts'][:, ref_idx[0]:ref_idx[1]], obj2world.unsqueeze(1)) #[B,n_ref,3,4]
    model_input = dict(cam_K=cam_K, cam_W=obj2cams,
                       image_ref=input_image,
                       verts=data['verts'],faces=data['faces'],
                       background=background, cube_diagonal=cube_diagonal,
                       cam_K_ref=cam_K_ref,cam_W_ref=cam_W_ref)
    model_input=to_device(model_input, device)
    return model_input

def split_into_batches(input_data,batch_size=8):
    batches=[]
    N=input_data['cam_K'].shape[1]
    for i in range(math.ceil(N/batch_size)):
        sub_bs=min(batch_size,N-i*batch_size)
        sub_input=dict()
        for k in input_data:
            if k=='n_views':
                sub_input[k]=sub_bs
            elif ('ref' in k) or ('constraint' in k) or k=='obj2world_normed' or k=='verts' or k=='faces':
                sub_input[k]=input_data[k]
            else:
                sub_input[k]=input_data[k][:,i*batch_size:i*batch_size+sub_bs]
        batches.append(sub_input)
    return batches


#https://github.com/pytorch/pytorch/issues/47562
def cal_gradient_penalty(model,real,fake):
    batch_size=real.shape[0]
    epsilon = torch.rand(batch_size,device=real.device,dtype=real.dtype)
    epsilon = epsilon.view(-1,1,1,1)
    inter = epsilon*real +(1-epsilon)*fake
    inter.requires_grad= True
    logits_inter = model(inter)
    gradient=autograd.grad(
        logits_inter,inter,torch.ones_like(logits_inter),
        create_graph=True,retain_graph=True,only_inputs=True
    )[0]
    gradient_penalty = (torch.linalg.norm(gradient.view(batch_size,-1),2,dim=1)-1)**2
    return gradient_penalty.mean()+0*logits_inter.mean()

def get_D_label_fake(masks, logit_shape):
    return F.interpolate(masks, logit_shape[2:], mode='nearest')

def random_flip_rotate(x,p=0.5):
    #x [3,sz,sz]
    if np.random.random()>p:
        return x
    k=np.random.randint(0,4)
    flip_dim=np.random.randint(0,3)
    x=torch.rot90(x,k,(1,2))
    if flip_dim!=0:
        x=torch.flip(x,(flip_dim,))
    return x

def random_flip_rotate_batch(patches,p=0.5):
    return torch.stack([random_flip_rotate(x,p=p) for x in patches],0)

class Trainer(BaseTrainer):
    '''
    Subclass of Basetrainer for defining train_step, eval_step and visualize
    '''

    def __init__(self, model_g, model_d,
                 optimizer_g, optimizer_d,
                 scheduler_g, scheduler_d,
                 renderer,

                 w_pix_ren=0,
                 w_pix_ref=0,
                 w_gan_ren=0,
                 w_gan_ref=0,
                 pix_ren_type='L1',
                 pix_ref_type='L1',
                 gan_type='standard',

                 multi_gpu=True,
                 crop_size= 128,
                 max_grad_norm=5,
                 lambda_gp=10,

                 n_ref=2,
                 n_render_train=3,
                 n_render_val=1,
                 transform_crop=True,
                 update_g_every=1,
                 **kwargs):

        # Initialize base trainer
        super().__init__(**kwargs)

        # Models and optimizers
        self.model_g = model_g
        self.model_d = model_d
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.max_grad_norm = max_grad_norm
        self.renderer = renderer
        
        self.w_pix_ren = w_pix_ren
        self.w_pix_ref = w_pix_ref
        self.w_gan_ren = w_gan_ren
        self.w_gan_ref = w_gan_ref

        self.pix_ren_type = pix_ren_type
        self.pix_ref_type = pix_ref_type
        self.gan_type = gan_type
        self.lambda_gp=lambda_gp

        self.perceptual_criterion=PerceptualLoss().to(self.device)
        # Checkpointer
        self.checkpoint_io.register_modules(
            model_g=self.model_g.module if multi_gpu else self.model_g, 
            model_d=self.model_d.module if multi_gpu else self.model_d,
            optimizer_g=self.optimizer_g,
        )

        if self.is_master:
            print('pixel type',(self.pix_ref_type,self.pix_ren_type),'gan type',self.gan_type)

        self.visualizer = TextureVisualizer(n_views=16,render_size=(128,128))

        if transform_crop:
            scales=(0.8, 1.25)
            ratios=(0.8, 1.25)
            offset=(-0.1,0.1)
        else:
            scales=(1.,1.)
            ratios=(1.,1.)
            offset=(0.,0.)

        self.crop_fake = RandomResizedCropAroundTarget((crop_size, crop_size), scale=scales, ratio=ratios,offset=offset)
        self.crop_true = RandomResizedCrop((crop_size, crop_size), scale=scales, ratio=ratios)
        self.crop_fake_center = RandomResizedCropAroundTarget((crop_size, crop_size), scale=(1,1), ratio=(1,1),
                                                              offset=(0., 0.))
        self.multi_gpu=multi_gpu

        self.n_ref=n_ref
        self.n_render_train=n_render_train
        self.n_render_val=n_render_val
        self.update_g_every=update_g_every # N steps Dis, 1 step generator
        
    def train_step(self, batch, epoch_it, it):
        '''
        A single training step for the conditional or generative experiment
        Output:
            Losses
        '''
        self.scheduler_g.step()
        self.scheduler_d.step()
        self.model_g.train()
        self.model_d.train()

        loss_dict=dict()
        if self.w_gan_ren>0:
            depth = self.renderer.render(batch,idx=0).squeeze(-1)  # [B,N_ref+N_render,H,W] all depth map
            depth_mask = (depth > 0).float().flatten(0,1).unsqueeze(1)  # [B*N_render,1,H,W] depth mask
            model_input= prepare_input(
                batch,
                device=self.device,
                render_idx=[0,self.n_ref+self.n_render_train], # views to render
                ref_idx=[0,self.n_ref])
            background= model_input['background'].clone().flatten(0,1).permute(0,3,1,2)  #[bs*n_render,3,h,w]
            loss_d = self.train_step_d(depth, depth_mask, model_input, background)
        else:
            loss_d = dict()

        #if True:
        if it%self.update_g_every==0:
            depth = self.renderer.render(batch,idx=0).squeeze(-1)  # [B,N_ref+N_render,H,W] all depth map
            depth_render=depth[:,self.n_ref:]
            depth_ref=depth[:,:self.n_ref]
            depth_mask_render = (depth_render > 0).float().flatten(0,1).unsqueeze(1)  # [B*N_render,1,H,W] depth mask
            depth_mask_ref = (depth_ref > 0).float().flatten(0, 1).unsqueeze(1) # [B*N_ref,1,H,W] depth mask
            model_input_render = prepare_input(
                batch,
                device=self.device,
                render_idx=[self.n_ref,self.n_ref+self.n_render_train],
                ref_idx=[0,self.n_ref])
            background_render = model_input_render['background'].clone().flatten(0,1).permute(0,3,1,2)  #[bs*n_render,3,h,w]
            model_input_ref = prepare_input(
                batch,
                device=self.device,
                render_idx=[0,self.n_ref],
                ref_idx=[0,self.n_ref]
            )
            background_ref = model_input_ref['background'].clone().flatten(0, 1).permute(0, 3, 1, 2)  # [bs*n_ref,3,h,w]
            loss_g = self.train_step_g(
                depth_render, depth_mask_render, model_input_render, background_render,
                depth_ref,depth_mask_ref,model_input_ref,background_ref)
        else:
            loss_g={}
        loss_dict.update(loss_d)
        loss_dict.update(loss_g)
        loss_dict['lr_g'] = self.optimizer_g.param_groups[0]['lr']
        loss_dict['lr_d'] = self.optimizer_d.param_groups[0]['lr']
        return loss_dict

    def set_grad(self, model, turn_on):
        for p in model.parameters():
            p.require_grads = turn_on

    def train_step_d(self, depth, depth_mask, model_input, background):
        '''
        A single train step of the discriminator
        '''
        self.set_grad(self.model_g, False)
        self.set_grad(self.model_d, True)
        self.optimizer_d.zero_grad()

        with torch.no_grad():
            fake_images = self.model_g(depth, **model_input,depth_ref=depth[:,:self.n_ref],visualize=False)
            fake_images=fake_images.permute(0,3,1,2)
            bboxes = get_bbox_from_mask(depth_mask)

            fake_crops=self.crop_fake(fake_images,bboxes)
            real_crops=self.crop_true(background)

            fake_crops=random_flip_rotate_batch(fake_crops)
            real_crops=random_flip_rotate_batch(real_crops)

            fake_crops=2*fake_crops-1
            real_crops=2*real_crops-1

        loss_dict=dict()
        logits_real = self.model_d(real_crops)
        logits_fake = self.model_d(fake_crops)
        if self.gan_type=='wgan-gp':
            gradient_penalty = cal_gradient_penalty(self.model_d, real_crops.detach(), fake_crops.detach())
            loss_d=logits_fake.mean()-logits_real.mean()+self.lambda_gp*gradient_penalty
            score_d= logits_real.mean()-logits_fake.mean()
        elif self.gan_type=='standard':
            loss_d=F.binary_cross_entropy_with_logits(logits_real,torch.ones_like(logits_real))\
                +F.binary_cross_entropy_with_logits(logits_fake,torch.zeros_like(logits_fake))
            score_d= 0.5*((logits_real>0).float().mean()+(logits_fake<=0).float().mean())
        elif self.gan_type=='ls':
            loss_d=F.mse_loss(logits_real,torch.ones_like(logits_real))\
                +F.mse_loss(logits_fake,torch.zeros_like(logits_fake))
            score_d= 0.5*((logits_real>0.5).float().mean()+(logits_fake<=0.5).float().mean())
        else:
            raise NotImplementedError("GAN type not implemented")
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(self.model_d.parameters(), self.max_grad_norm)
        self.optimizer_d.step()
        loss_dict['loss_d']=to_float(loss_d)
        loss_dict['score_d']=to_float(score_d)
        return loss_dict

    def train_step_g(self, depth_render, depth_mask_render, model_input_render, background_render,
                            depth_ref,depth_mask_ref,model_input_ref,background_ref):
        '''
        A single train step of the generator part of generative model 
        (VAE: Encoder+Decoder and GAN: Generator)
        '''

        self.set_grad(self.model_g, True)
        self.set_grad(self.model_d, False)
        self.optimizer_g.zero_grad()
        #torch.autograd.set_detect_anomaly(True)
        #use gradient accumulation to reduce memory usage
        loss_dict=dict()
        if self.w_pix_ren>0 or self.w_gan_ren>0:
            fake_images = self.model_g(
                depth_render, **model_input_render,depth_ref=depth_ref,visualize=False
            )
            fake_images=fake_images.permute(0,3,1,2)    
            loss_gan_total=0.
            loss_pix_total=0.
            if self.w_gan_ren>0:
                #crop results and scale results
                bboxes = get_bbox_from_mask(depth_mask_render)
                fake_crops=self.crop_fake(fake_images,bboxes)
                fake_crops=2*fake_crops-1
                logits=self.model_d(fake_crops)
                if self.gan_type=='wgan-gp':
                    loss_gan=-logits.mean()
                elif self.gan_type=='standard':
                    loss_gan=F.binary_cross_entropy_with_logits(logits,torch.ones_like(logits))
                elif self.gan_type=='ls':
                    loss_gan=F.mse_loss(logits,torch.ones_like(logits))
                else:
                    raise NotImplementedError("GAN type not implemented")
                loss_gan_total+=self.w_gan_ren*loss_gan
                assert not torch.isnan(loss_gan), f"Nan loss detected in gan loss"
                loss_dict['loss_gan_ren']=to_float(loss_gan)

            if self.w_pix_ren>0:
                loss_pix=self.compute_pix_loss(fake_images,background_render,depth_mask_render,self.pix_ren_type)
                loss_dict[f'loss_pix_render']=to_float(loss_pix)
                loss_pix_total+=loss_pix*self.w_pix_ren

            total_loss=loss_gan_total+loss_pix_total
            total_loss.backward()
            #loss_dict['loss_gan_total']=to_float(loss_gan_total)
            #loss_dict['loss_pix_total']=to_float(loss_pix_total)
            loss_dict['loss_generator_ren']=to_float(total_loss)

        if self.w_pix_ref>0 or self.w_gan_ref>0:
            fake_images= self.model_g(
                depth_ref, **model_input_ref,depth_ref=depth_ref
            )
            fake_images=fake_images.permute(0,3,1,2)    
            loss_gan_total=0.
            loss_pix_total=0.
            if self.w_gan_ref>0:
                #crop results and scale results
                bboxes = get_bbox_from_mask(depth_mask_ref)
                fake_crops=self.crop_fake(fake_images,bboxes)
                fake_crops=2*fake_crops-1
                logits=self.model_d(fake_crops)
                if self.gan_type=='wgan-gp':
                    loss_gan=-logits.mean()
                elif self.gan_type=='standard':
                    loss_gan=F.binary_cross_entropy_with_logits(logits,torch.ones_like(logits))
                elif self.gan_type=='ls':
                    loss_gan=F.mse_loss(logits,torch.ones_like(logits))
                else:
                    raise NotImplementedError("GAN type not implemented")
                loss_gan_total+=self.w_gan_ref*loss_gan
                assert not torch.isnan(loss_gan), f"Nan loss detected in gan loss"
                loss_dict['loss_gan_ref']=to_float(loss_gan)

            if self.w_pix_ref>0:
                loss_pix=self.compute_pix_loss(fake_images,background_ref,depth_mask_ref,self.pix_ref_type)
                loss_dict[f'loss_pix_ref']=to_float(loss_pix)
                loss_pix_total+=loss_pix*self.w_pix_ref
            total_loss=loss_gan_total+loss_pix_total
            total_loss.backward()
            loss_dict['loss_generator_ref']=to_float(total_loss)
        if self.max_grad_norm>0:
            torch.nn.utils.clip_grad_norm_(self.model_g.parameters(), self.max_grad_norm)
        self.optimizer_g.step()
        return loss_dict

    def compute_pix_loss(self, img_fake, img_real, masks,loss_type):
        '''
        Compute Pixelloss
        '''
        if loss_type == 'L2':
            loss = ((img_fake-img_real)**2 * masks).sum()
            loss /= masks.sum()
        elif loss_type == 'L1':
            # loss = F.l1_loss(img_fake, img_real)
            loss = (torch.abs(img_fake - img_real) * masks).sum()
            loss /= masks.sum()
        elif loss_type == 'perceptual':
            bboxes=get_bbox_from_mask(masks)
            #print(img_fake.shape,img_real.shape)
            fake_crops = self.crop_fake(torch.cat([img_fake, img_real], dim=1), bboxes)
            fake_fg= fake_crops[:,:3]
            fake_bg= fake_crops[:,3:]
            loss=self.perceptual_criterion(fake_fg,fake_bg)            
        return loss

    def get_vis_data(self, loader):
        vis_bs = 4
        vis_data = []
        for i in range(vis_bs):
            data = loader.dataset.get_all_views(i,self.n_ref,val_only=True)
            data = collate_fn([data])
            vis_data.append(data)
        return vis_data

    @torch.no_grad()
    def visualize(self, vis_data, it, phase,safe_batchsize=8):
        self.model_g.eval()
        if self.multi_gpu:
            vis_model=self.model_g.module
        else:
            vis_model=self.model_g
        os.makedirs(f"{self.vis_dir}/iter_{phase}_{it}/", exist_ok=True)
        for i, batch in enumerate(vis_data):
            assert len(batch['verts']) == 1,"For visualization batchsize can only be 1"
            depth = self.renderer.render(batch,idx=0).squeeze(-1)  # [B,N_views,H,W] depth map
            model_input = prepare_input(
                batch, 
                device=self.device,
                render_idx=[0,depth.shape[1]],
                ref_idx=[0,self.n_ref])
            model_input['depth']=depth
            depth_ref=depth[:,:self.n_ref]
            safe_batches=split_into_batches(model_input,safe_batchsize)

            rendered_images_all=[]
            for safe_batch in safe_batches:
                rendered_images = vis_model(**safe_batch,depth_ref=depth_ref)
                rendered_images=rendered_images.detach().cpu().numpy()[:,:,:,::-1] #[b,h,w,c]
                rendered_images_all.append(rendered_images)
            rendered_images_all=np.concatenate(rendered_images_all,0)
            N = rendered_images_all.shape[0]
            for j in range(N):
                cv2.imwrite(
                        f"{self.vis_dir}/iter_{phase}_{it}/sample_{i}_{batch['image_meta'][0][j]}.png",
                        rendered_images_all[j] * 255
                )
            #visualize fake views
            mesh = Meshes(verts=batch['verts'], faces=batch['faces']).to(self.device)
            background = torch.ones(*self.visualizer.background_shape).unsqueeze(0).to(self.device)
            N = background.shape[1]

            input_image = model_input['image_ref']
            cam_K_ref = model_input['cam_K_ref']
            cam_W_ref = model_input['cam_W_ref']
            cube_diagonal = batch['cube_diagonal'].repeat(N).view(1,-1).to(self.device)
            depth=self.visualizer.render_all_views(mesh).squeeze(-1)

            rendered_fake=[]
            for k in range(math.ceil(N / safe_batchsize)):
                sub_bs = min(safe_batchsize, N - k * safe_batchsize)
                p_3d, p_2d, mask = self.visualizer.depth_map_to_3d(
                    depth[k*safe_batchsize:k*safe_batchsize+sub_bs],
                    k*safe_batchsize,k*safe_batchsize+sub_bs)
                rendered_images = vis_model(None,None,None,
                                            image_ref=input_image,
                                            background=background[:,k*safe_batchsize:k*safe_batchsize+sub_bs],
                                            cube_diagonal=cube_diagonal[:,k*safe_batchsize:k*safe_batchsize+sub_bs],
                                            cam_K_ref=cam_K_ref,
                                            cam_W_ref=cam_W_ref,
                                            verts=model_input['verts'],faces=model_input['faces'],
                                            p_3d=p_3d,p_2d=p_2d,pad_mask=mask,
                                            depth_ref=depth_ref,visualize=False
                                            )
                rendered_images=rendered_images.detach().cpu().numpy()
                rendered_fake.append(rendered_images)
            rendered_fake = np.concatenate(rendered_fake,0) 
            plt.figure(figsize=(20, 20))
            for j in range(16):
                plt.subplot(4, 4, j + 1)
                plt.imshow(rendered_fake[j])
            plt.tight_layout()
            plt.savefig(f"{self.vis_dir}/iter_{phase}_{it}/sample_{i}_fake_views.png")
            plt.close()
            