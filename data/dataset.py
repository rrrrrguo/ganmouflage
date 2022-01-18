from data.scene import Scene
import os
import numpy as np
from pytorch3d.io import load_objs_as_meshes, load_obj
import pandas as pd
from glob import glob
import torch
from torch.utils.data import Subset
import random

def estimate_face_normals(verts,faces):
    pt1=verts[faces[:,0]]
    pt2=verts[faces[:,1]]
    pt3=verts[faces[:,2]] #[nface,3]
    n = np.cross(pt2- pt1,pt3- pt1) #[nface,3]
    n = n/ (np.linalg.norm(n,2,-1,keepdims=True)+1e-7) #[A,B,C] Ax+By+Cz+D=0
    #find D
    D= -(n*pt1).sum(-1)
    return n,D

def get_dataset_single_scene(
    cfg, debug=False, 
    scene_name="walden-tree3",
    fake_cube=True,animals=True,
    val_samples=400,cube_name=None):
    scene_folder = cfg['data']['scene']['scene_folder']
    n_views_ref = cfg['data']['scene']['n_views_ref']
    n_views_sup = cfg['data']['scene']['n_views_sup']
    n_views_val = cfg['data']['scene']['n_views_val']
    target_size = cfg['data']['scene']['target_size']
    distance_to_reference = cfg['data']['scene']['distance_to_cube']
    cube_scale = cfg['data']['scene']['cube_scale']
    if scene_name=='esp-tree':
        cube_save_type=2
    elif scene_name in ['bookshelf-real','couch5-real','patio2-real']:
        cube_save_type=1
    else:
        cube_save_type=0
    scene = [Scene(
        scene_folder + f'{scene_name}/', target_size,
        n_views_ref=n_views_ref, n_views_sup=n_views_sup,n_views_val=n_views_val,
        distance_to_reference=distance_to_reference, debug=debug, 
        cube_scale=cube_scale,cube_save_type=cube_save_type,cube_name=cube_name
        )]
    if fake_cube:
        datasets = {
            'train': CamoDataset_Fake(
                scene, cfg['data']['shape']['train_rot_limit'],
                val=False,
                train_cube_scale_range=cfg['data']['train_cube_scale_range']),
            'valid': CamoDataset_Fake(scene, [0, 0, 0],val=True),
        }
        datasets['valid'].repeat=val_samples
    elif animals:
        obj_list=glob(cfg['data']['animals']['data_dir']+'*.obj')
        obj_list=sorted(obj_list)
        datasets = {
            'train': CamoDataset_Animals(
                obj_list,scene, cfg['data']['shape']['train_rot_limit'],
                val=False,train_cube_scale_range=cfg['data']['train_cube_scale_range'],
                repeat=50),
            'valid': CamoDataset_Animals(
                obj_list,scene, [0, 0, 0],
                val=True,repeat=1),
        }
    else:
        raise NotImplementedError("Unsupported dataset")
    return datasets


class CamoDataset_Fake:
    def __init__(self, scenes, rot_limit=None,val=False,same_views=False,train_cube_scale_range=[1,1]):
        self.scenes = scenes
        print("Using scenes:")
        for scene in self.scenes:
            print(scene.scene_dir)

        if rot_limit is not None:
            self.rot_limit = np.array(rot_limit, dtype=np.float32)
        else:
            self.rot_limit = None
        self.verts, self.faces, _ = load_obj("fake_cube/model_simplified.obj", load_textures=False)
        self.verts, self.faces, _ = load_obj("fake_chair/chair.obj", load_textures=False)
        Ry = np.array([np.cos(150/180*np.pi), 0, -np.sin(150/180*np.pi),
                       0, 1, 0,
                       np.sin(150/180*np.pi), 0, np.cos(150/180*np.pi)]).reshape(3, 3)
        self.verts= self.verts@Ry.T
        self.verts=self.verts.float()
        self.val=val
        self.same_views=same_views
        self.train_cube_scale_range=train_cube_scale_range
        self.repeat=2000

    def gen_random_rot(self, rot_limit=None):
        '''
        Parameters
        ----------
        limit  Rotation limit in x,y,z direction

        Returns
        -------
        R    3x3 rotation matrix
        '''
        angles = np.random.uniform(-1, 1, 3)
        if rot_limit is None:
            angles *= np.pi
        else:
            angles *= rot_limit/180*np.pi
        Rx = np.array([1, 0, 0,
                       0, np.cos(angles[0]), -np.sin(angles[0]),
                       0, np.sin(angles[0]), np.cos(angles[0])]).reshape(3, 3)
        Ry = np.array([np.cos(angles[1]), 0, -np.sin(angles[1]),
                       0, 1, 0,
                       np.sin(angles[1]), 0, np.cos(angles[1])]).reshape(3, 3)
        Rz = np.array([np.cos(angles[2]), -np.sin(angles[2]), 0,
                       np.sin(angles[2]), np.cos(angles[2]), 0,
                       0, 0, 1]).reshape(3, 3)
        R = Rz @ Ry @ Rx
        return R

    def __getitem__(self, idx):
        # random select scene
        scene_idx = np.random.randint(0, len(self.scenes))
        scene_parameters = self.scenes[scene_idx].generate_random_scene(
            val=self.val)
        data = dict()
        data.update(scene_parameters)
        if not self.val:
            cube_scale=np.random.uniform(*self.train_cube_scale_range)
            data['verts']= self.verts*cube_scale
        else:
            data['verts'] = self.verts
        data['faces'] = self.faces.verts_idx
        data['obj_rotation'] = self.gen_random_rot(self.rot_limit).astype(np.float32) @ data['obj2worldR']
        data['obj_id'] = idx
        return data

    def get_all_views(self, idx,n_ref,val_only=False,ref_views=[],fixed_position=False):
        # random select scene
        scene_idx = idx % len(self.scenes)
        if idx==0 or fixed_position:
            distance=0
            rot_limit=np.array([0,0,0])
        else:
            distance=None
            rot_limit=self.rot_limit
        scene_parameters = self.scenes[scene_idx].get_all_views(
            distance=distance,
            val_only=val_only,
            n_ref=n_ref,ref_views=ref_views)
        data = dict()
        data.update(scene_parameters)

        data['verts'] = self.verts
        data['faces'] = self.faces.verts_idx
        data['obj_rotation'] = self.gen_random_rot(rot_limit).astype(np.float32) @ data['obj2worldR']
        return data

    def __len__(self):
        return self.repeat


class CamoDataset_Animals:
    def __init__(self, object_list, scenes, rot_limit=None,val=False,same_views=False,train_cube_scale_range=[1,1],repeat=50):
        self.scenes = scenes
        print("Using scenes:")
        for scene in self.scenes:
            print(scene.scene_dir)
        self.val=val
        self.object_list = object_list*repeat
        if rot_limit is not None:
            self.rot_limit = np.array(rot_limit, dtype=np.float32)
        else:
            self.rot_limit = None
        self.same_views=same_views
        self.train_cube_scale_range=train_cube_scale_range

    def gen_random_rot(self, rot_limit=None):
        '''
        Parameters
        ----------
        limit  Rotation limit in x,y,z direction

        Returns
        -------
        R    3x3 rotation matrix
        '''
        angles = np.random.uniform(-1, 1, 3)
        if rot_limit is None:
            angles *= np.pi
        else:
            angles *= rot_limit/180*np.pi
        Rx = np.array([1, 0, 0,
                       0, np.cos(angles[0]), -np.sin(angles[0]),
                       0, np.sin(angles[0]), np.cos(angles[0])]).reshape(3, 3)
        Ry = np.array([np.cos(angles[1]), 0, -np.sin(angles[1]),
                       0, 1, 0,
                       np.sin(angles[1]), 0, np.cos(angles[1])]).reshape(3, 3)
        Rz = np.array([np.cos(angles[2]), -np.sin(angles[2]), 0,
                       np.sin(angles[2]), np.cos(angles[2]), 0,
                       0, 0, 1]).reshape(3, 3)
        R = Rz @ Ry @ Rx
        return R

    def __getitem__(self, idx):
        # load obj
        obj_path = self.object_list[idx]
        verts, faces, _ = load_obj(obj_path, load_textures=False)
        scene_idx = np.random.randint(0, len(self.scenes))
        scene_parameters = self.scenes[scene_idx].generate_random_scene(val=self.val)

        data = dict()
        data.update(scene_parameters)
        if not self.val:
            cube_scale=np.random.uniform(*self.train_cube_scale_range)
            data['verts']= verts*cube_scale
        else:
            data['verts'] = verts
        data['faces'] = faces.verts_idx
        data['obj_rotation'] = self.gen_random_rot(self.rot_limit).astype(np.float32) @ data['obj2worldR']
        data['obj_id'] = obj_path.split("/")[-1]
        return data

    def get_all_views(self, idx,n_ref,val_only=False,ref_views=[],fixed_position=False):
        obj_path = self.object_list[idx]
        #obj_path = self.root_dir + f"{model_id}/model_simplified.obj"
        pc = np.load(obj_path.replace(".obj",".npy"))
        verts, faces, _ = load_obj(obj_path, load_textures=False)
        scene_idx = idx % len(self.scenes)
        if idx==0 or fixed_position:
            distance=0
            rot_limit=np.array([0,0,0])
        else:
            distance=None
            rot_limit=self.rot_limit
        scene_parameters = self.scenes[scene_idx].get_all_views(
            distance=distance,
            val_only=val_only,
            n_ref=n_ref,ref_views=ref_views)
        data = dict()
        data.update(scene_parameters)
        data['verts'] = verts
        data['faces'] = faces.verts_idx
        data['obj_rotation'] = self.gen_random_rot(rot_limit).astype(np.float32) @ data['obj2worldR']
        data['obj_id'] = obj_path.split("/")[-1]
        return data

    def __len__(self):
        return len(self.object_list)

