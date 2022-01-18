import torch

def split_to_list(input, index, bs):
    result = []
    for i in range(bs):
        result.append(input[index == i])
    return result

def cat_one(x):
    '''
    Args:
        x: tensor, ...,3
    
    Returns:
        tensor, ...,4
    '''
    ones=torch.ones((*x.shape[:-1],1),dtype=x.dtype,device=x.device)
    return torch.cat([x,ones],dim=-1)

def proj2image(points, cam_K, cam_W):
    '''
    Args:
        points: Batch,N_render, K, 3
        cam_K: Batch, N_ref, 3,3
        cam_W: Batch, N_ref, 3,4

    Returns:
        2d coord: [N,N_ref,N_render,K,3],
        depth: [N,N_ref,N_render,K,1]
    '''
    P=torch.matmul(cam_K, cam_W)
    points = cat_one(points) #[N,N_render,K,4]
    points = torch.matmul(P.unsqueeze(2), points.unsqueeze(1).transpose(-2,-1)).transpose(-2,-1)# [N,N_ref,N_render,K,3]
    return points[..., :2] / points[..., 2:3], points[..., 2:3]


def normalize_p2d(p_2d, H, W):
    p_2d[...,0]= 2*p_2d[...,0]/W -1
    p_2d[...,1]= 2*p_2d[...,1]/H -1
    return p_2d

def get_ray_directions_inv(p_2d, cam_K_ref):
    '''
    Args:
        p_2d:    [bs,n_ref,n_render,K,2]  projected coordinates
        cam_K_ref: [bs,n_ref,3,3] projection matrix

    Returns:
        directions: [bs,n_ref,n_render,K,3]
    '''
    cam_K_inv = torch.linalg.inv(cam_K_ref)

    p = cat_one(p_2d)
    directions = torch.matmul(cam_K_inv.unsqueeze(
        2), p.transpose(-2, -1)).transpose(-2, -1)
    directions /= directions.norm(p=2, dim=-1, keepdim=True)
    return directions
