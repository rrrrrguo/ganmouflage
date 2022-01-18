import os
from data.utils import collate_fn
from data.dataset import get_dataset_single_scene
from data.render_depth import DepthRenderer
import matplotlib.pyplot as plt
import torch
from model_v4.build_models import get_models
import yaml
import cv2
from trainer.training import prepare_input,split_into_batches
import numpy as np
from pytorch3d.structures import Meshes
from model_v4.visualizer import TextureVisualizer
import math
import random
import cv2
import argparse

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED=1234
seed_everything(SEED)

def visualize_depth(depth):
    mask = depth > 0
    if mask.astype(float).sum() == 0:
        return mask
    min_val = depth[mask].min()
    max_val = depth[mask].max()
    depth[mask] = (depth[mask] - min_val) / (max_val - min_val)
    return depth

def generate_shift(max_shift=0.25,size=5,sigma=1):
    k=cv2.getGaussianKernel(size,sigma)
    k=k@k.T
    grid_shift=np.linspace(-max_shift,max_shift,size)
    shift_2d=np.stack(np.meshgrid(grid_shift,grid_shift),-1).reshape(-1,2) #[size^2,2]
    weight=k.reshape(-1)
    return shift_2d.astype(np.float32),weight.astype(np.float32)

def get_cube_world_position(cube_diagonal,rot,translation):
    fake_cube_vertices=np.array([[-0.5, -0.5, -0.5],
                                 [-0.5, -0.5, 0.5],
                                 [0.5, -0.5, -0.5],
                                 [0.5, -0.5, 0.5],
                                 [-0.5, 0.5, -0.5],
                                 [-0.5, 0.5, 0.5],
                                 [0.5, 0.5, -0.5],
                                 [0.5, 0.5, 0.5],
                                 ])/np.sqrt(3)*cube_diagonal
    return (rot@fake_cube_vertices.T).T+translation

if __name__ == '__main__':
    safe_batchsize=32
    parser = argparse.ArgumentParser(
        description='Generate camflagued textures for objects.'
    )
    parser.add_argument('--model_path', type=str, help='Path to model weights(folder).')
    parser.add_argument('--out_path',type=str,default='result/test_run',help="Path to save rendered images")
    parser.add_argument('--use_aa',action='store_true')
    parser.add_argument('--n',type=int,default=5,help="Number of samples to generate, \
        sample 0 will be fixed at the predefined place, other samples will be randomly sampled nearby the predefined place")
    parser.add_argument("--save_background",action='store_true',help="Whether to save background images")
    parser.add_argument("--animals",type=int,nargs='+',default=[], help="Shape ID for animal shapes, if not specified all animal shapes will be used.")
    args=parser.parse_args()

    model_dir=str(args.model_path)
    cfg_path=f"{model_dir}/config.yaml"
    ckpt_path=f"{model_dir}/model.pt"
    out_dir=str(args.out_path)

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    shift_max=0.25
    grid_sz=5
    sigma=1

    os.makedirs(out_dir,exist_ok=True)
    print(yaml.dump(cfg))
    datasets = get_dataset_single_scene(
        cfg, debug=False,scene_name=cfg['data']['scene']['scene_name'],
        fake_cube=cfg['data']['fake_cube']
    )
    is_cube=cfg['data']['fake_cube']
    dataset=datasets['valid']
    device='cuda'
    model=get_models(cfg,device)
    model=model['generator']
    renderer = DepthRenderer(cfg['data']['scene']['target_size'])
    checkpoint=torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_g'])
    visualizer=TextureVisualizer(16,(192,192))
    n_ref=cfg['data']['scene']['n_views_ref']
    if not is_cube:
        animals=args.animals if args.animals else np.arange(len(dataset))
        print(len(animals),"animals")
    else:
        animals=[0]
    for i in range(args.n):
        for animal_idx in animals:
            os.makedirs(out_dir+f"/sample_{i}/", exist_ok=True)
            data = dataset.get_all_views(animal_idx+len(dataset)*i,n_ref=n_ref,val_only=False,fixed_position=i==0)
            cube_world_positions=get_cube_world_position(dataset.scenes[0].cube_diagonal,data['obj2worldR'],data['obj2world'])
            np.save(out_dir+f"/sample_{i}/cube_positions.npy",cube_world_positions)
            data = collate_fn([data])
            assert len(data['verts']) == 1, "For visualization batchsize can only be 1"

            depth = renderer.render(data,idx=0).squeeze(-1)
            depth_ref=depth[:,:n_ref]

            model_input = prepare_input(
                data, 
                device=device,
                render_idx=[0,depth.shape[1]],
                ref_idx=[0,n_ref])
            model_input['depth']=depth
            safe_batches=split_into_batches(model_input,safe_batchsize)
            rendered_images_all=[]
            for safe_batch in safe_batches:
                rendered_images = model(**safe_batch,depth_ref=depth_ref)
                rendered_images=rendered_images.detach().cpu().numpy()[:,:,:,::-1] #[b,h,w,c]
                rendered_images_all.append(rendered_images)
            rendered_images_all=np.concatenate(rendered_images_all,0)
            N = rendered_images_all.shape[0]

            for j in range(N):
                cv2.imwrite(f"{out_dir}/sample_{i}/{data['image_meta'][0][j]}_shape_{animal_idx}_mask.png",depth[0][j].detach().cpu().numpy()* 255)
                cv2.imwrite(
                    f"{out_dir}/sample_{i}/{data['image_meta'][0][j]}_shape_{animal_idx}.png",
                    rendered_images_all[j] * 255
                )
            if i==0 and args.save_background and animal_idx==0:
                os.makedirs(f"{out_dir}/background/",exist_ok=True)
                for j in range(N):
                    meta=data['image_meta'][0][j]
                    if 'ref' in meta:
                        continue
                    view_id=meta.split('_')[1]
                    cv2.imwrite(
                        f"{out_dir}/background/background_view{view_id}.png",
                        model_input['background'][0][j].detach().cpu().numpy()[:,:,::-1] * 255
                    )
            #visualize fake views
            mesh = Meshes(verts=data['verts'], faces=data['faces']).to(device)
            background = torch.ones(*visualizer.background_shape).unsqueeze(0).to(device)
            N = background.shape[1]
            input_image = model_input['image_ref']
            cam_K_ref = model_input['cam_K_ref']
            cam_W_ref = model_input['cam_W_ref']
            cube_diagonal = data['cube_diagonal'].repeat(N).view(1,-1).to(device)
            depth=visualizer.render_all_views(mesh).squeeze(-1)
            rendered_fake=[]
            for k in range(math.ceil(N / safe_batchsize)):
                sub_bs = min(safe_batchsize, N - k * safe_batchsize)
                p_3d, p_2d, mask = visualizer.depth_map_to_3d(depth[k*safe_batchsize:k*safe_batchsize+sub_bs],idx_start=k*safe_batchsize,idx_end=k*safe_batchsize+sub_bs)
                rendered_images = model(None,None,None,
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
            plt.figure(figsize=(24, 24))
            for j in range(16):
                plt.subplot(4, 4, j + 1)
                plt.imshow(rendered_fake[j])
            plt.tight_layout()
            plt.savefig(f"{out_dir}/sample_{i}/fake_views_shape_{animal_idx}.png")
            plt.close()