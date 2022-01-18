import math
import os
import numpy as np
from data.utils import (get_sample_boundary,get_scene_parameters_olddata,
                        plane_estimation,
                        sample_position,
                        project_to_image,
                        load_background_image_aa,
                        cameraToNDC, draw_gaussian_mask)
import cv2
from sklearn.metrics import pairwise_distances
import pickle
from sklearn.model_selection import train_test_split

BOX_EDGES=[[0,1],[1,3],[2,3],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]
def draw_cube(image,vertices,color):
    #pairwise_distances=pairwise_distances(self.cube_vertices)
    #cube_diagonal=pairwise_distances.max()
    #edges= np.where(pairwise_distances<cube_diagonal/math.sqrt(3))
    verts_int=vertices.astype(np.int)
    for edge in BOX_EDGES:
        pt1=verts_int[edge[0]]
        pt2=verts_int[edge[1]]
        image=cv2.line(image,pt1,pt2,color,1)
    return image


class Scene:
    def __init__(
        self, scene_dir, target_size, 
        n_views_ref=2, n_views_sup=3,n_views_val=1,
        distance_to_reference=3,cube_scale=1., 
        debug=False,cache_images=True,
        cube_save_type=0,cube_name=None):

        self.scene_dir = scene_dir

        self.n_views_ref=n_views_ref
        self.n_views_sup=n_views_sup
        self.n_views_val=n_views_val
        self.focals, self.Rts, self.view_fns, self.cube_vertices,self.cubeR = get_scene_parameters_olddata(scene_dir,cube_save_type=cube_save_type,cube_name=cube_name)
        # find full shape
        self.num_frames=len(self.focals)
        assert self.num_frames>=n_views_ref+n_views_sup+n_views_val, "not enough views in the scene"

        first_im = cv2.imread(self.scene_dir + self.view_fns[0])
        self.full_shape = first_im.shape
        self.scale = (target_size[0] / self.full_shape[0], target_size[1] / self.full_shape[1])
        self.target_size = target_size

        self.cube_center = self.cube_vertices.mean(0)
        self.cube_vertices = (self.cube_vertices - self.cube_center) * cube_scale + self.cube_center
        self.cube_diagonal=pairwise_distances(self.cube_vertices).max()

        self.distance_to_reference = distance_to_reference * self.cube_diagonal/math.sqrt(3)
        print(scene_dir)
        print("Cube scale", cube_scale)
        print("Cube Diagonal", self.cube_diagonal)
        # get the buttom 4 vertices for the cube to fit a ground plane
        buttom_vertices = self.cube_vertices[:4]
        self.plane = plane_estimation(buttom_vertices)
        self.buttom_vertices = buttom_vertices

        if cache_images:
            print("Caching scene images")
            self.image_cached=[
                load_background_image_aa(
                    self.scene_dir + self.view_fns[idx],
                    (self.target_size[1], self.target_size[0])) for idx in range(len(self.focals))]
        else:
            self.image_cached=None

        if os.path.isfile(scene_dir+'/train_val_v3.pkl'):
            with open(scene_dir+'/train_val_v3.pkl','rb') as f:
                self.train_views,self.val_views=pickle.load(f)
        else:
            raise FileNotFoundError("Train test split not found")
        print("Train view idx",self.train_views)
        print("Valid view idx",self.val_views)
        self.debug = debug

    def get_target_size(self):
        return self.target_size

    def get_P(self, f):
        return self.get_K(f) @ self.get_Rt(f)  # K@[R|T]

    def get_K(self, f):
        K = np.array([[-self.scale[1] * self.focals[f], 0., -0.5 + self.scale[1] * self.full_shape[1] / 2.],
                      [0., self.scale[0] * self.focals[f], -0.5 + self.scale[0] * self.full_shape[0] / 2.],
                      [0., 0., 1.]])
        return K

    def get_Rt(self, f):
        R = self.Rts[f][0]
        t = self.Rts[f][1]
        return np.concatenate([R, t[:, None]], axis=1)

    def verify_position(self, cube_new_vertices,view_idx):
        ''''
            Find the views that include the object
        '''
        good_views = []
        for idx in view_idx:
            box = project_to_image(self.get_P(idx), cube_new_vertices)  # [8,2] vertiecs [x,y]
            good_points=(box[:,0]< self.target_size[1]) & (box[:,1]< self.target_size[0]) & (box[:,0]>=0) & (box[:,1]>=0)
            if np.sum(good_points.astype(np.float))>=5:
                good_views.append(idx)
        return good_views

    def generate_random_scene(self,val=False):
        '''
        Args:
            val: if true, choose n view from validation views
        '''
        if val:
            good_views_idx = []
            good_views_idx_val= []
            cube_new_vertices = self.cube_vertices
            obj2world = self.cube_center
            while len(good_views_idx) < self.n_views_ref or len(good_views_idx_val)<self.n_views_val:
                obj2world = sample_position(self.plane, self.cube_center, self.distance_to_reference)
                cube_new_vertices = self.cube_vertices - self.cube_center + obj2world
                good_views_idx = self.verify_position(cube_new_vertices,self.train_views)
                good_views_idx_val=self.verify_position(cube_new_vertices,self.val_views)
            chosen_views_ref=np.random.choice(good_views_idx, self.n_views_ref,replace=False)
            chosen_views_val=np.random.choice(good_views_idx_val,self.n_views_val,replace=False)
            chosen_views=np.concatenate([chosen_views_ref,chosen_views_val],0)
        else:
            good_views_idx = []
            cube_new_vertices = self.cube_vertices
            obj2world = self.cube_center
            n_views=self.n_views_ref+self.n_views_sup
            while len(good_views_idx) < n_views:
                obj2world = sample_position(self.plane, self.cube_center, self.distance_to_reference)
                cube_new_vertices = self.cube_vertices - self.cube_center + obj2world
                good_views_idx = self.verify_position(cube_new_vertices,self.train_views)
            chosen_views=np.random.choice(good_views_idx, n_views,replace=False)

        scene_images = []
        scene_masks = []
        scene_Rts = []
        focal_NDC = []
        proj_NDC = []
        cam_K = []

        for idx in chosen_views:
            # for idx in [0,1]:
            scene_Rts.append(self.get_Rt(idx))
            box = project_to_image(self.get_P(idx), cube_new_vertices)
            box_dim = box.max(0) - box.min(0)  # [xdim,ydim]
            box_center = (box.max(0) + box.min(0)) / 2  # [xcoord,ycoord]
            if self.image_cached is not None:
                image = self.image_cached[idx]
            else:
                image = load_background_image_aa(self.scene_dir + self.view_fns[idx],
                                          (self.target_size[1], self.target_size[0]))

            if self.debug:
                for v in box:
                    image = cv2.circle(image, (int(v[0]), int(v[1])), 3, (0, 1, 0))
                for v in project_to_image(self.get_P(idx), self.cube_vertices):
                    image = cv2.circle(image, (int(v[0]), int(v[1])), 3, (1, 0, 0))
                image=draw_cube(image,box,(0,1,0))
                image=draw_cube(image,project_to_image(self.get_P(idx), self.cube_vertices),(1,0,0))
                sample_bound=get_sample_boundary(self.plane, self.cube_center, self.distance_to_reference)
                projected_bound=project_to_image(self.get_P(idx), sample_bound)
                image=cv2.polylines(image,projected_bound.astype(np.int),True,(0,0,1),2)
                for v in project_to_image(self.get_P(idx), self.buttom_vertices):
                    image = cv2.circle(image, (int(v[0]), int(v[1])), 5, (0, 0, 1))

            fx, fy, px, py = cameraToNDC(self.focals[idx],
                                         self.full_shape[1] / 2 - 0.5, self.full_shape[0] / 2 - 0.5,
                                         self.full_shape[1], self.full_shape[0])
            scene_images.append(image)
            scene_masks.append(draw_gaussian_mask(self.target_size, box_center, box_dim))
            focal_NDC.append([fx, fy])
            proj_NDC.append([px, py])
            cam_K.append(self.get_K(idx))

        cam_K = np.stack(cam_K, 0)
        scene_images = np.stack(scene_images, 0)
        scene_Rts = np.stack(scene_Rts, 0)
        scene_masks = np.stack(scene_masks, 0)
        return {
            'scene_images': scene_images.astype(np.float32),  # [N_views,H,W,3]
            'scene_masks': scene_masks.astype(np.float32),  # [N_views,H,W]
            'scene_Rts': scene_Rts.astype(np.float32),  # [N_views,3,4]
            'obj2world': obj2world.astype(np.float32),  # [3]
            'obj2worldR': self.cubeR.astype(np.float32), #rotation matrix
            'focal_NDC': np.array(focal_NDC, dtype=np.float32),  # [N_views,2]
            'proj_NDC': np.array(proj_NDC, dtype=np.float32),  # [N_views,2]
            'cam_K': cam_K.astype(np.float32),
            'cube_diagonal':self.cube_diagonal.astype(np.float32),  # Diagonal length,used to scale the object
        }

    def get_all_views(self,distance=None,val_only=False,n_ref=1,ref_views=[]):
        '''for visualization use
        Args:
            distance: will override the distance parameter in the scene
            val: if true, reference images will be selected from validation views
            n_ref: number of reference images
        Returns:

        '''
        good_views_idx = []
        cube_new_vertices=self.cube_vertices

        while len(good_views_idx) < n_ref:
            obj2world = sample_position(self.plane, self.cube_center, distance if distance is not None else self.distance_to_reference)
            cube_new_vertices = self.cube_vertices - self.cube_center + obj2world
            good_views_idx=self.train_views
            good_views_idx = self.verify_position(cube_new_vertices,self.train_views)

        reference_idx=np.random.choice(good_views_idx, n_ref,replace=False)
        reference_idx=(ref_views + [x for x in reference_idx if x not in ref_views])[:n_ref]
        print("Using refernce views",reference_idx)

        image_meta=[f'ref_{idx}' for idx in reference_idx]
        views=[reference_idx]
        if not val_only:
            to_vis_views=np.concatenate([self.train_views,self.val_views],0)
        else:
            to_vis_views=self.val_views
        to_vis_views=np.sort(to_vis_views)
        for idx in to_vis_views:
            image_meta.append(f'render_{idx}_train' if idx in self.train_views else f'render_{idx}_val')
        views.append(to_vis_views)
        image_idxs=np.concatenate(views,0)

        scene_images = []
        scene_masks = []
        scene_Rts = []
        focal_NDC = []
        proj_NDC = []
        cam_K = []

        for idx in image_idxs:
            scene_Rts.append(self.get_Rt(idx))
            box = project_to_image(self.get_P(idx), cube_new_vertices)
            box_dim = box.max(0) - box.min(0)  # [xdim,ydim]
            box_center = (box.max(0) + box.min(0)) / 2  # [xcoord,ycoord]
            image = load_background_image_aa(self.scene_dir + self.view_fns[idx],
                                          (self.target_size[1], self.target_size[0]))

            if self.debug:
                sample_bound=get_sample_boundary(self.plane, self.cube_center, self.distance_to_reference)
                projected_bound=project_to_image(self.get_P(idx), sample_bound)
                image=cv2.polylines(image,[projected_bound.astype(np.int)],True,(0.6,0.8,1),1)
                image=draw_cube(image,project_to_image(self.get_P(idx), self.cube_vertices),(1,0,0)) #old cube position
                image=draw_cube(image,box,(0,1,0)) #refernce position

            fx, fy, px, py = cameraToNDC(self.focals[idx],
                                         self.full_shape[1] / 2 - 0.5, self.full_shape[0] / 2 - 0.5,
                                         self.full_shape[1], self.full_shape[0])
            scene_images.append(image)
            scene_masks.append(draw_gaussian_mask(self.target_size, box_center, box_dim))
            focal_NDC.append([fx, fy])
            proj_NDC.append([px, py])
            cam_K.append(self.get_K(idx))

        cam_K = np.stack(cam_K, 0)
        scene_images = np.stack(scene_images, 0)
        scene_Rts = np.stack(scene_Rts, 0)
        scene_masks = np.stack(scene_masks, 0)
        return {
            'scene_images': scene_images.astype(np.float32),  # [N_views,H,W,3]
            'scene_masks': scene_masks.astype(np.float32),  # [N_views,H,W]
            'scene_Rts': scene_Rts.astype(np.float32),  # [N_views,3,4]
            'obj2world': obj2world.astype(np.float32),  # [3] translation
            'obj2worldR': self.cubeR.astype(np.float32), #rotation
            'focal_NDC': np.array(focal_NDC, dtype=np.float32),  # [N_views,2]
            'proj_NDC': np.array(proj_NDC, dtype=np.float32),  # [N_views,2]
            'cam_K': cam_K.astype(np.float32),
            'cube_diagonal': self.cube_diagonal.astype(np.float32),  # Diagonal length
            'image_meta':image_meta
        }
