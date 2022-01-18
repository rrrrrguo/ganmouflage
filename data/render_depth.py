from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    PerspectiveCameras,
)
from pytorch3d.structures import Meshes
import torch
import torch.nn as nn

def toMatrix(R,t):
    '''
    Parameters
    ----------
    R         [N,3,3]
    t         [N,3]

    Returns
    -------
    Rt        [N,4,4]
    '''
    assert R.shape[0]==t.shape[0], "R,t dimension mismatch"
    N=R.shape[0]
    Rt=torch.cat([R,t.view(-1,3,1)],dim=2) #[N,3,4]
    Rt=torch.cat([Rt,torch.tensor([0,0,0,1]).view(1,1,4).repeat(N,1,1)],dim=1) #[N,4,4]
    return Rt


class RastWarper(nn.Module):
    def __init__(self,rast):
        super().__init__()
        self.rast=rast
    
    def forward(self,meshes,cameras):
        return self.rast(meshes,cameras=cameras).zbuf


class DepthRenderer:
    def __init__(self,render_size,device='cuda',axis_mirror=[-1,1,-1]):
        print("render size",render_size)
        hard_raster_settings = RasterizationSettings(
            image_size=render_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )
        self.rasterizer= MeshRasterizer(
                cameras=None,
                raster_settings=hard_raster_settings
            )
        self.rasterizer=RastWarper(self.rasterizer)
        self.device=device
        self.axis_mirror_matrix=torch.diag(torch.tensor(axis_mirror)).unsqueeze(0).float()

    def build_cameras(self,data,idx,idx_end):
        focal_NDC=data['focal_NDC'][:,idx:idx_end].reshape(-1,2)
        proj_NDC=data['proj_NDC'][:,idx:idx_end].reshape(-1,2)
        obj2world=toMatrix(data['obj_rotation'],data['obj2world']) #[N,4,4]
        world2camera=data['scene_Rts'][:,idx:idx_end] #[N,nview,3,4]
        n_views=world2camera.shape[1]
        obj2world=obj2world.unsqueeze(1).repeat(1,n_views,1,1).reshape(-1,4,4)
        obj2cams=torch.bmm(world2camera.reshape(-1,3,4),obj2world)#[N*nview,3,4]
        obj2cams_with_mirror=torch.bmm(self.axis_mirror_matrix.repeat(obj2cams.shape[0],1,1),obj2cams)
        cameras=PerspectiveCameras(focal_length=focal_NDC,
                                  principal_point=proj_NDC,
                                  R=obj2cams_with_mirror[:,:,:3].permute(0,2,1),
                                  T=obj2cams_with_mirror[:,:,3],
                                  device=self.device)
        return cameras

    @torch.no_grad()
    def render(self,data,idx=0,idx_end=None):
        '''
        Parameters
        ----------
        data       Batched data

        Returns
        -------
        depth      Rendered depth
        '''
        #scale object according to diagonal length
        meshes=Meshes(verts=[v*d for v,d in zip(data['verts'],data['cube_diagonal'])],faces=data['faces'])
        cameras=self.build_cameras(data,idx,idx_end)
        if idx<0:
            idx = data['focal_NDC'].shape[1]+idx
        if idx_end is None:
            idx_end=data['focal_NDC'].shape[1]  #render to the last image
        n_views=idx_end-idx
        meshes=meshes.extend(n_views).to(self.device)
        fragments=self.rasterizer(meshes,cameras=cameras)
        depth=fragments
        return depth.view(-1,n_views,*depth.shape[1:])

