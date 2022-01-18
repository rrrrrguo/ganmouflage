import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet_unet import bilinear_interpolation, bilinear_interpolation_list
from .utils import *
from .sample_normals import sample_normals_v2
from torchvision.transforms.functional import affine
from torchvision.transforms import InterpolationMode
from .decoder import get_nerf_embedder
try:
    import open3d
except ImportError:
    print("Warning! open3d not installed, open3d visualization will fail")

def extract_projected_features(points,feature_maps, cam_K_ref, cam_W_ref,size):
    """Project points to input views and extract feature vector through bilinear interpolation
    Args:
        points ([torch.Tensor]): Point 3D coordinate in  [bs,K,3] or [bs,N_render,K,3] format
        feature_maps ([List[torch.Tensor]]): List of image feature maps in [bs,ch,h,w] format
        cam_K_ref ([torch.Tensor]): Camera intrinsic matrix in [bs,N_ref,3,3] format 
        cam_W_ref ([torch.Tensor]): Camera extrinsic matrix in [bs,N_ref,3,4] format
        size ([tuple]): Image size (H,W)
    Return:
        projected_features: Extracted features in [bs,n_ref,K,ch] or [bs,n_ref,n_render,K,ch] format 
        depth: Depth value in [bs,n_ref,K,1] or [bs,n_ref,n_render,K,1] format 
    """

    assert points.shape[0]==cam_K_ref.shape[0]==cam_W_ref.shape[0]
    bs=points.shape[0]
    n_ref=cam_K_ref.shape[1]
    K=points.shape[-2]
    if points.dim()==3:
        # in bs,k,3 format
        squeeze=True
        points=points.unsqueeze(1) #[bs,1,K,3]
        n_render=1
    else:
        squeeze=False
        n_render=points.shape[1]
    points_2d,depth=proj2image(points,cam_K_ref,cam_W_ref) #[bs,nref,N_render,K,2]
    points_2d_normed = normalize_p2d(points_2d.clone(),size[0],size[1]) #[bs,nref,N_render,K,2]
    projected_features = bilinear_interpolation_list(
        feature_maps, points_2d_normed.flatten(2,3).flatten(0, 1),mode='bilinear')  # [bs*nref,ch,N_render*K]
    projected_features=projected_features.view(bs,n_ref,-1,n_render,K).permute(0,1,3,4,2)
    if squeeze:
        projected_features=projected_features.squeeze(2)
        depth=depth.squeeze(2)
        points_2d=points_2d.squeeze(2)
    return projected_features,depth,points_2d

def dilate_depth_boundary(depth,k=5):
    #depth mask
    depth=depth.unsqueeze(1) #[bs*nref,1,H,W]
    depth_mask=(depth>0).float()
    padding=(k-1)//2
    dilated=F.max_pool2d(depth,k,1,padding)
    return dilated*(1-depth_mask)+depth*depth_mask #[bs*nref,1,H,W]

@torch.no_grad()
def sample_relative_depth(points,depth_ref,cam_K_ref, cam_W_ref,size):
    """Project points to input views and find its relative depth value,  depth/min_depth at that point
    Args:
        points ([torch.Tensor]): Point 3D coordinate in  [bs,K,3] or [bs,N_render,K,3] format
        depth_ref ([torch.Tensor]): Depth map for reference views, [bs,1,H,W]
        cam_K_ref ([torch.Tensor]): Camera intrinsic matrix in [bs,N_ref,3,3] format 
        cam_W_ref ([torch.Tensor]): Camera extrinsic matrix in [bs,N_ref,3,4] format
        size ([tuple]): Image size (H,W)
    Return:
        min_depth: Min depth value at that projected location
        depth: Depth value in [bs,n_ref,K,1] or [bs,n_ref,n_render,K,1] format 
    """
    #first dilate the depth map to avoid problems near boundary
    dilated_depth=dilate_depth_boundary(depth_ref)
    assert points.shape[0]==cam_K_ref.shape[0]==cam_W_ref.shape[0]
    bs=points.shape[0]
    n_ref=cam_K_ref.shape[1]
    K=points.shape[-2]
    if points.dim()==3:
        # in bs,k,3 format
        #squeeze=True
        points=points.unsqueeze(1) #[bs,1,K,3]
        n_render=1
        squeeze=True
    else:
        n_render=points.shape[1]
        squeeze=False
    points_2d,depth=proj2image(points,cam_K_ref,cam_W_ref) #[bs,nref,N_render,K,2]
    points_2d_normed = normalize_p2d(points_2d.clone(),size[0],size[1]) #[bs,nref,N_render,K,2]
    min_depth = bilinear_interpolation(
        dilated_depth, points_2d_normed.flatten(2,3).flatten(0, 1))  # [bs*nref,ch,N_render*K]
    min_depth=min_depth.view(bs,n_ref,-1,n_render,K).permute(0,1,3,4,2)
    if squeeze:
        min_depth=min_depth.squeeze(2)
        depth=depth.squeeze(2)
    return -depth,min_depth


class TextureNetwork(nn.Module):
    def __init__(
        self,
        image_encoder,
        decoder_pixel,
        cat_relative_depth=False,
        cat_surface_normals=False,
        ):
        '''
        Args:
            decoder:            Decoder that returns the color of given location
            geometry_encoder:   3D geometric encoder
            image_encoder:      2D scene image encoder
                                (assume unet encoder, use bilinear interpolation to extract local features)
        '''
        super().__init__()
        self.image_encoder = image_encoder
        self.decoder_pixel = decoder_pixel
        self.cat_relative_depth=cat_relative_depth
        self.cat_surface_normals=cat_surface_normals
        self.nerf_emb,_=get_nerf_embedder(6,1)

    @torch.no_grad()
    def depth_map_to_3d(self, depth, cam_Ks, cam_Ws, shift_2d=None):
        """Derive 3D locations of each foreground pixel of a depth map.

        Args:
            depth (torch.FloatTensor): tensor of size B x N x M
                with depth at every pixel (-1 is background, >0 foreground)
            cam_Ks (torch.FloatTensor): tensor of size B x 3 x 3 representing
                camera matrices
            cam_Ws (torch.FloatTensor): tensor of size B x 3 x 4 representing
                camera from obj space matrices
            shift: None or tuple [2] shifting value(x_shift,y_shift) for 2d coordinate, used for anti-aliasing
        Returns:
            p_3d (torch.FloatTensor):       3d coordinates in object space, already padded to [bs,k,3]
            p_2d (tuple[torch.IntTensor]):  2d coordinates of foreground pixels, can be directly used for indexing in background image
            p_mask (torch.FloatTensor):     pad mask for p_3d/p_2d, 0 is the padded value
        """
        assert len(depth.shape) == 3
        batch_size = depth.shape[0]
        device = depth.device
        # find foreground depth values and their index
        batch_index, y_2d, x_2d = torch.nonzero(
            depth > 0, as_tuple=True)  # [K,3]
        p_2d = (batch_index, y_2d.clone(), x_2d.clone())

        x_2d = x_2d.float()
        y_2d = y_2d.float()
        if shift_2d is not None:
            x_2d += shift_2d[0]
            y_2d += shift_2d[1]
            translation = (-shift_2d[0], -shift_2d[1])
            depth = affine(depth, 0, translation, 1.0,
                           (0., 0.), InterpolationMode.BILINEAR)
        foreground_depth_values = -depth[p_2d].unsqueeze(1)

        p_screen = torch.stack(
            [x_2d, y_2d, torch.ones_like(x_2d, device=device)], dim=1).float()
        p_screen *= foreground_depth_values  # [K,3] [zx,zy,z]
        # convert to a list of tensors List[torch.FloatTensor[K',3]]
        p_screen = split_to_list(p_screen, batch_index, batch_size)
        # convert then pad
        # calculate inv matrices
        inv_Ks = torch.inverse(cam_Ks)
        zero_one_row = torch.tensor([[0., 0., 0., 1.]])
        zero_one_row = zero_one_row.expand(batch_size, 1, 4).to(device)
        cam_Ws = torch.cat((cam_Ws, zero_one_row), dim=1)
        inv_Ws = torch.inverse(cam_Ws)
        # calculate p_3d
        p_3d = []
        for p, inv_K, inv_W in zip(p_screen, inv_Ks, inv_Ws):
            p_camera = inv_K @ p.T
            p_camera_homog = torch.cat([p_camera, torch.ones(
                (1, p_camera.shape[1]), device=device)], dim=0)  # [4,k]
            p_3d_homog = inv_W @ p_camera_homog  # [4,k]
            p_3d.append(p_3d_homog[:3, :].T)
        # get a padding mask, 1 shows the value is valid
        p_mask = torch.nn.utils.rnn.pad_sequence([torch.ones(x.shape[0], device=device) for x in p_3d],
                                                 batch_first=True, padding_value=0.0)  # [bs,K]
        p_3d = torch.nn.utils.rnn.pad_sequence(
            p_3d, batch_first=True, padding_value=0.0)  # [bs,K,3]
        return p_3d, p_2d, p_mask

    def forward(self, depth,
                cam_K, cam_W,
                image_ref, 
                verts,faces,
                background, cube_diagonal,
                cam_K_ref=None, cam_W_ref=None,
                depth_ref=None,
                visualize=False,
                p_3d=None, p_2d=None, pad_mask=None,
                shift_2d=None,
                ):
        """Generate an image from rendered depth map.

        Args:
            depth (torch.FloatTensor): tensor of size B x n_render x N x M
                representing depth of at pixels
            cam_K (torch.FloatTensor): tensor of size B x n_render x 3 x 3 representing
                camera projection matrix
            cam_W (torch.FloatTensor): tensor of size B x n_render x 3 x 4 representing
                camera from object space matrix
            image_ref (torch.FloatTensor): tensor of size B x n_ref x 3 x H x W representing scene images
            verts (List(torch.FloatTensor)): List of tensor of N_verts x 3
            faces (List(torch.FloatTensor)): List of tensor of N_faces x 3 
            background (torch.FloatTensor): tensor of size B x n_render x  3 x H x W representing target images 
            cube_diagonal (torch.FloatTensor): tensor of size B representing cube diagonal length, 
                which is used to scale object during rendering
            cam_K_ref (torch.FloatTensor): tensor of size B x n_ref x 3 x 3 representing
                camera projectin matrix (for image ref)
            cam_W_ref (torch.FloatTensor): tensor of size B x n_ref x 3 x 4 representing
                camera from object space matrix (for image ref)
            visualize: for debugging, show visible points and point cloud
            shift_2d: Used for anti-aliasing
            p_3d,p_2d,pad_mask: used for skip depthmap to p3d
        Returns:
            img (torch.FloatTensor): tensor of size B x 3 x H x W representing
                output image
        """
        n_render = background.shape[1]
        n_ref = image_ref.shape[1]
        batch_size = background.shape[0]
        assert (cam_K_ref.shape[-2:] == (3, 3))
        assert (cam_W_ref.shape[-2:] == (3, 4))
        if p_3d is None:
            assert (cam_K.shape[-2:] == (3, 3))
            assert (cam_W.shape[-2:] == (3, 4))
            p_3d, p_2d, pad_mask = self.depth_map_to_3d(depth.flatten(
                0, 1), cam_K.flatten(0, 1), cam_W.flatten(0, 1), shift_2d=shift_2d)
            # we scale the object in rendering, so we unscale here
            p_3d /= cube_diagonal.view(-1, 1, 1)  # [bs*n_render,K,3]

        x=p_3d.clone()
        # visualize depth inverse rendering and point clouds
        if visualize:
            for i in range(batch_size):
                for j in range(n_render):
                    points=p_3d[i*n_render+j]
                    m=pad_mask[i*n_render+j]
                    points = points[m.bool()]
                    pdc = open3d.geometry.PointCloud()
                    points = points.cpu().numpy()
                    pdc.points = open3d.utility.Vector3dVector(points)
                    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.2, origin=[0,0,0])
                    
                    v=verts[i].cpu().numpy()
                    f=faces[i].cpu().numpy()
                    obj_mesh= open3d.geometry.TriangleMesh(
                        vertices=open3d.utility.Vector3dVector(v),
                        triangles=open3d.utility.Vector3iVector(f))
                    mesh_wire=open3d.geometry.LineSet.create_from_triangle_mesh(obj_mesh)
                    open3d.visualization.draw_geometries([pdc,mesh_frame,mesh_wire])
                    #open3d.io.write_point_cloud(f"vis_intermediate/sample_{i}_view{j}_visiblepoints.ply",pdc, write_ascii=True)
                    #open3d.io.write_line_set(f"vis_intermediate/sample_{i}_view{j}_lineset.ply",mesh_wire, write_ascii=True)
        # list[[bs*n_ref,ch,H,W]], feature map in different resolution
        feature_map = self.image_encoder(image_ref.flatten(0, 1))
        #get pixel aligned features 
        pixel_aligned_features,p_3d_depth,p_2d_unnormed=extract_projected_features(
            (p_3d*cube_diagonal.view(-1, 1, 1)).unflatten(0, [batch_size, n_render]),
            feature_map,cam_K_ref, cam_W_ref,
            (image_ref.shape[3],image_ref.shape[4])
        ) #[bs,n_ref,n_render,K,ch]
        #print(pixel_aligned_features.shape)
        pixel_aligned_features=pixel_aligned_features.permute(0,2,1,3,4).flatten(0,1)
        z=[pixel_aligned_features]

        if self.cat_relative_depth:
            assert depth_ref is not None
            relative_depth,min_depth=sample_relative_depth(
            (p_3d*cube_diagonal.view(-1, 1, 1)).unflatten(0, [batch_size, n_render]),
            depth_ref.flatten(0,1),cam_K_ref, cam_W_ref,
            (image_ref.shape[3],image_ref.shape[4])
            ) 
            #[bs,n_ref,n_render,K,1]
            relative_depth=relative_depth.permute(0,2,1,3,4).flatten(0,1)
            min_depth=min_depth.permute(0,2,1,3,4).flatten(0,1)
            z.append(self.nerf_emb(relative_depth))
            z.append(self.nerf_emb(min_depth))

        if self.cat_surface_normals:
            surface_normals = sample_normals_v2(verts,faces, p_3d.unflatten(
                0, [batch_size, n_render]),visualize=visualize)
            surface_normals = surface_normals.unsqueeze(1)# [bs,1,nrender,K,3]
            # print(surface_normals.shape)
            surface_normals = torch.matmul(cam_W_ref[:, :, None, :, :3], surface_normals.transpose(
                -2, -1)).transpose(-2, -1)  # [bs,nref,nrender,K,3]
            # print(surface_normals.shape)
            ray_directions = get_ray_directions_inv(p_2d_unnormed, cam_K_ref)
            if visualize:
                for i in range(batch_size):
                    for j in range(n_render):
                        points=p_3d[i*n_render+j]
                        m=pad_mask[i*n_render+j]
                        points = points[m.bool()]

                        pdc = open3d.geometry.PointCloud()
                        points=torch.matmul(cam_W_ref[i,0], cat_one(points).T).T #[k,3]
                        points = points.cpu().numpy()
                        normals = surface_normals[i,0,j].cpu().numpy()
                        pdc.points = open3d.utility.Vector3dVector(points)
                        pdc.normals = open3d.utility.Vector3dVector(normals)
                        v=torch.matmul(cam_W_ref[i,0], cat_one(verts[i]).T).T.cpu().numpy() 
                        f=faces[i].cpu().numpy()
                        obj_mesh= open3d.geometry.TriangleMesh(
                            vertices=open3d.utility.Vector3dVector(v),
                            triangles=open3d.utility.Vector3iVector(f))
                        mesh_wire=open3d.geometry.LineSet.create_from_triangle_mesh(obj_mesh)
                        open3d.visualization.draw_geometries([pdc,mesh_wire])
            z.append(surface_normals.permute(0,2,1,3,4).flatten(0,1))
            z.append(ray_directions.permute(0,2,1,3,4).flatten(0,1))

        z=torch.cat(z,-1)
        colors=self.decoder_pixel(x,z) 

        # mask is [bs,k],get foreground colors
        colors = colors[pad_mask.bool()]
        rendered_fine = background.clone().flatten(0, 1)
        rendered_fine[p_2d] = colors  # insert result
        return rendered_fine
