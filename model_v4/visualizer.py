from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    look_at_view_transform,
    FoVPerspectiveCameras
)
from pytorch3d.io import load_objs_as_meshes
import torch
import matplotlib.pyplot as plt
from .modules.utils import split_to_list
from data.render_depth import toMatrix
# Generate 16 images going a tour around the object
def flip_y_angle(R):
    x = torch.atan2(R[:,2, 1], R[:,2, 2])
    y = torch.atan2(-R[:,2, 0], torch.sqrt(R[:,2, 1] * R[:,2, 1] + R[:,2, 2] * R[:,2, 2]))
    z = torch.atan2(R[:,1, 0], R[:,0, 0])

    y*=-1
    x*=-1
    z*=1

    Rx=torch.zeros_like(R)
    Ry=torch.zeros_like(R)
    Rz=torch.zeros_like(R)

    Rx[:,0,0] = 1.
    Rx[:,1,1] = torch.cos(x)
    Rx[:,1,2] = -torch.sin(x)
    Rx[:,2,1] = torch.sin(x)
    Rx[:,2,2] = torch.cos(x)

    Ry[:,1,1] = 1.
    Ry[:,0,0] = torch.cos(y)
    Ry[:,0,2] = torch.sin(y)
    Ry[:,2,0] = -torch.sin(y)
    Ry[:,2,2] = torch.cos(y)

    Rz[:,2,2]= 1.
    Rz[:,0,0] = torch.cos(z)
    Rz[:,0,1] = -torch.sin(z)
    Rz[:,1,0] = torch.sin(z)
    Rz[:,1,1] = torch.cos(z)

    return torch.bmm(torch.bmm(Rz,Ry),Rx)

class TextureVisualizer():
    def __init__(self, n_views=20, render_size=(384, 576), device='cuda'):
        # Get a batch of viewing angles.
        self.n_views = n_views
        elev = torch.linspace(30,30, n_views)
        azim = torch.linspace(0, 180, n_views)
        R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T,)
        hard_raster_settings = RasterizationSettings(
            image_size=render_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )
        self.rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=hard_raster_settings
        )
        m = cameras.get_projection_transform().get_matrix()[0]
        fx = m[0, 0] * render_size[1] / 2
        fy = m[1, 1] * render_size[0] / 2
        px = render_size[1] / 2
        py = render_size[0] / 2
        K = [[fx, 0, px],
             [0, fy, py],
             [0, 0, 1]]
        self.Ks=torch.tensor(K).view(1,3,3).repeat(n_views,1,1).to(device)
        R2=flip_y_angle(R.permute(0,2,1))
        self.Ws=toMatrix(R2,T).to(device)
        self.background_shape=[n_views,render_size[0],render_size[1],3]


    def render_all_views(self, mesh):
        assert len(mesh) == 1
        meshes = mesh.extend(self.n_views)
        fragments = self.rasterizer(meshes)
        return fragments.zbuf  # [n_view,H,W,1]

    @torch.no_grad()
    def depth_map_to_3d(self, depth,idx_start,idx_end):
        '''
        Args:
            depth (torch.FloatTensor): tensor of size B x N x M
                with depth at every pixel (-1 is background, >0 foreground)

        Returns:
            p_3d (torch.FloatTensor):       3d coordinates in object space, already padded to [bs,k,3]
            p_2d (tuple[torch.IntTensor]):  2d coordinates of foreground pixels, can be directly used for indexing
            p_mask (torch.FloatTensor):     pad mask for p_3d, 0 is the padded value
        '''
        assert len(depth.shape) == 3
        batch_size = depth.shape[0]
        device = depth.device
        # find foreground depth values and their index
        batch_index, y_2d, x_2d = torch.nonzero(depth > 0, as_tuple=True)  # [K,3]
        foreground_depth_values = depth[(batch_index, y_2d, x_2d)].unsqueeze(1)  # [K,1]
        p_2d = (batch_index, y_2d, x_2d)

        p_screen = torch.stack([x_2d, y_2d, torch.ones_like(x_2d, device=device)], dim=1).float()
        p_screen *= foreground_depth_values  # [K,3] [zx,zy,z]
        # convert to a list of tensors List[torch.FloatTensor[K',3]]
        p_screen = split_to_list(p_screen, batch_index, batch_size)
        # convert then pad
        # calculate inv matrices
        inv_Ks = torch.inverse(self.Ks)[idx_start:idx_end]
        inv_Ws = torch.inverse(self.Ws)[idx_start:idx_end]
        assert len(inv_Ks)==batch_size
        # calculate p_3d
        p_3d = []
        for p, inv_K, inv_W in zip(p_screen, inv_Ks, inv_Ws):
            p_camera = inv_K @ p.T
            p_camera_homog = torch.cat([p_camera, torch.ones((1, p_camera.shape[1]), device=device)], dim=0)  # [4,k]
            #p_camera_homog[1,:]*=-1
            p_3d_homog = inv_W @ p_camera_homog  # [4,k]
            p_3d.append(p_3d_homog[:3, :].T)
        # get a padding mask, 1 shows the value is valid
        p_mask = torch.nn.utils.rnn.pad_sequence([torch.ones(x.shape[0], device=device) for x in p_3d],
                                                 batch_first=True, padding_value=0.0)  # [bs,K]
        p_3d = torch.nn.utils.rnn.pad_sequence(p_3d, batch_first=True, padding_value=0.0)  # [bs,K,3]
        p_3d[:,:,1]*=-1
        #p_3d[:,:,2]*=-1
        p_3d[:,:,0]*=-1
        return p_3d, p_2d, p_mask

    def get_p3d(self,mesh):
        depth = self.render_all_views(mesh).squeeze(-1)
        p_3d, p_2d, mask = self.depth_map_to_3d(depth,0,len(depth))
        return p_3d,p_2d,mask
