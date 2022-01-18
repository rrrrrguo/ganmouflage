import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet_unet import bilinear_interpolation, bilinear_interpolation_list
from .utils import *
from .sample_normals import sample_normals_v2
from torchvision.transforms.functional import affine
from torchvision.transforms import InterpolationMode
from trimesh.ray.ray_pyembree import RayMeshIntersector
from trimesh import Trimesh
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



class TextureNetworkBaseline(nn.Module):
    def __init__(
        self,
        method='mean',
        ):
        '''
        Args:
            decoder:            Decoder that returns the color of given location
            geometry_encoder:   3D geometric encoder
            image_encoder:      2D scene image encoder
                                (assume unet encoder, use bilinear interpolation to extract local features)
        '''
        super().__init__()
        self.method=method

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
            shift_2d: None or tuple [2] shifting value(x_shift,y_shift) for 2d coordinate, used for anti-aliasing
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

    @torch.no_grad()
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
        feature_map = [image_ref.flatten(0, 1)]
        #get pixel aligned features 
        pixel_aligned_features,p_2d_depth,p_2d_unnormed=extract_projected_features(
            (p_3d*cube_diagonal.view(-1, 1, 1)).unflatten(0, [batch_size, n_render]),
            feature_map,cam_K_ref, cam_W_ref,
            (image_ref.shape[3],image_ref.shape[4])
        ) #[bs,n_ref,n_render,K,ch]
        if self.method=='mean':
            colors=pixel_aligned_features.mean(1).flatten(0,1) #[bs*nrender,K,3]
        elif self.method=='greedy_pixel':
            surface_normals = sample_normals_v2(verts,faces, p_3d.unflatten(
                0, [batch_size, n_render]),visualize=visualize)
            surface_normals = surface_normals.unsqueeze(1)# [bs,1,nrender,K,3]
            # print(surface_normals.shape)
            surface_normals = torch.matmul(cam_W_ref[:, :, None, :, :3], surface_normals.transpose(
                -2, -1)).transpose(-2, -1)  # [bs,nref,nrender,K,3]
            # print(surface_normals.shape)
            ray_directions = get_ray_directions_inv(p_2d_unnormed, cam_K_ref) # [bs,nref,nrender,K,3]
            dots= torch.abs((surface_normals*ray_directions).sum(-1)) #[bs,nref,nrender,K]
            dots= F.one_hot(torch.argmax(dots,dim=1),num_classes=n_ref).permute(0,3,1,2)
            colors=(pixel_aligned_features*dots.unsqueeze(-1)).sum(dim=1).flatten(0,1)

        elif self.method=='greedy_pixel_weighted':
            surface_normals = sample_normals_v2(verts,faces, p_3d.unflatten(
                0, [batch_size, n_render]),visualize=visualize)
            surface_normals = surface_normals.unsqueeze(1)# [bs,1,nrender,K,3]
            # print(surface_normals.shape)
            surface_normals = torch.matmul(cam_W_ref[:, :, None, :, :3], surface_normals.transpose(
                -2, -1)).transpose(-2, -1)  # [bs,nref,nrender,K,3]
            # print(surface_normals.shape)
            ray_directions = get_ray_directions_inv(p_2d_unnormed, cam_K_ref) # [bs,nref,nrender,K,3]
            dots= torch.abs((surface_normals*ray_directions).sum(-1)) #[bs,nref,nrender,K]
            dots/=dots.sum(1,keepdim=True)
            colors=(pixel_aligned_features*dots.unsqueeze(-1)).sum(dim=1).flatten(0,1)
        
        elif self.method in ['random','greedy']:
            pixel_aligned_features=pixel_aligned_features.flatten(2,3)
            colors=[]
            for b in range(batch_size):
                if self.method=='random':
                    seq=list(range(0,n_ref)) #ref view order is already shuffled in the scene generation
                else:
                    vert=verts[b]
                    face=faces[b]
                    pt1=vert[face[:,0]]
                    pt2=vert[face[:,1]]
                    pt3=vert[face[:,2]] #[nface,3]
                    n = torch.cross(pt2- pt1,pt3- pt1,dim=1) #[nface,3]
                    n = F.normalize(n,2,1) #[nface,3]
                    
                    face_centers= (pt1+pt2+pt3)/3 #[nface,3]
                    face_centers = cat_one(face_centers) #[nface,4]
                    face_centers_in_camera = torch.matmul(cam_W_ref[b], face_centers.transpose(-2,-1)).transpose(-2,-1)# [n_ref,nface,3]
                    ray_directions_in_camera= F.normalize(face_centers_in_camera,p=2,dim=-1)
                    surface_normals_in_camera= torch.matmul(cam_W_ref[b,:,:,:3],n.unsqueeze(0).transpose(-2,-1)).transpose(-2,-1) #[n_ref,nface,3]
                    cosine= torch.abs((ray_directions_in_camera*surface_normals_in_camera).sum(-1)) #[n_ref,n_face]
                    scores_ref_views=(cosine>np.cos(20/180*np.pi)).float().sum(1)
                    seq=torch.argsort(scores_ref_views,descending=True).cpu().numpy()
                    print(scores_ref_views,seq)
                
                color=torch.zeros(
                    (pixel_aligned_features.shape[2],3),
                    dtype=torch.float,device=pixel_aligned_features.device
                )
                assigned=torch.zeros(
                    (pixel_aligned_features.shape[2]),
                    device=pixel_aligned_features.device
                ).bool()
                for i in range(n_ref):
                    v=seq[i]
                    verts_in_camera= (cam_W_ref[b,v] @ cat_one(cube_diagonal[b,0]*verts[b]).T).T  #[n,3]
                    faces_in_camera=faces[b]
                    mesh=Trimesh(verts_in_camera.cpu().numpy(),faces_in_camera.cpu().numpy())
                    p_2d_query=p_2d_unnormed[b,v].flatten(0,1)  #[n_render*k,2]
                    actual_depth=-p_2d_depth[b,v].reshape(-1) #[n_render*k]
                    ray_origins=np.zeros((p_2d_query.shape[0],3))
                    ray_directions=torch.matmul(torch.linalg.inv(cam_K_ref[b,v]),cat_one(p_2d_query).T).T
                    ray_directions=F.normalize(ray_directions,2,-1)
                    #ray_directions[:,-1]*=-1
                    ray_mesh=RayMeshIntersector(mesh)
                    _,index_ray,intersect_loc=ray_mesh.intersects_id(ray_origins,-ray_directions.cpu().numpy(),multiple_hits=False,return_locations=True)
                    #print(mesh,locations,ray_origins.shape,ray_directions.shape)
                    nearest_depth=np.zeros((len(p_2d_query)))
                    nearest_depth[index_ray]=intersect_loc[:,2]
                    nearest_depth=-torch.tensor(nearest_depth,device=actual_depth.device,dtype=actual_depth.dtype)

                    normalized_delta= (actual_depth-nearest_depth)/cube_diagonal[b,0]
                    #plt.subplot(batch_size,n_ref,1+v+n_ref*b)
                    #plt.hist(normalized_delta.cpu().numpy(),bins=np.linspace(-0.2,0.5,25))
                    visible=torch.logical_and(normalized_delta<0.015,nearest_depth!=0)
                    to_assign= torch.logical_and(visible,~assigned)
                    print("View",v,"Assigning", to_assign.int().sum().item())
                    assigned=torch.logical_or(to_assign,assigned)
                    color[to_assign]=pixel_aligned_features[b,v][to_assign]
                color[~assigned]=pixel_aligned_features[b,seq[-1]][~assigned]
                colors.append(color)
                #print((1-assigned.float()).sum().item(),"not assigned")
                #print((1-pad_mask.view(batch_size,n_render,-1)[b].float()).sum().item(),"is padded")
            colors=torch.stack(colors,0)
            #plt.savefig("relative_depth.png")
            colors=colors.view(batch_size*n_render,-1,3)

        colors = colors[pad_mask.bool()]
        rendered_fine = background.clone().flatten(0, 1)
        rendered_fine[p_2d] = colors  #insert result
        return rendered_fine
