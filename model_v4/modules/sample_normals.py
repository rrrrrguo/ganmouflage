import torch
import torch.nn.functional as F
from pytorch3d import _C
try:
    import open3d
except ImportError:
    print("Warning! open3d not installed, open3d visualization will fail")

@torch.no_grad()
def sample_normals_v2(verts,faces,p_3d,visualize=False):
    '''
    Args:
        face_normals: List[Tensor[n_v,3]]
        face_Ds: List[Tensor[n_f,3]]
        p_3d: [bs,n_render,K,3]

    Returns:
    '''
    normals=[]
    bs,n_render,K,_=p_3d.shape
    for vert,face,p in zip(verts,faces,p_3d):
        pt1=vert[face[:,0]]
        pt2=vert[face[:,1]]
        pt3=vert[face[:,2]] #[nface,3]

        n = torch.cross(pt2- pt1,pt3- pt1,dim=1) #[nface,3]
        n = F.normalize(n,2,1) #[nface,3] #[A,B,C] Ax+By+Cz+D=0

        #find D
        obj_mesh= open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(vert.cpu().numpy()),
            triangles=open3d.utility.Vector3iVector(face.cpu().numpy()))
        mesh_wire=open3d.geometry.LineSet.create_from_triangle_mesh(obj_mesh)

        dists, face_idx = _C.point_face_dist_forward(
            p.flatten(0,1), 
            torch.tensor([0],dtype=torch.long,device=p.device), 
            torch.stack([pt1,pt2,pt3],dim=1), 
            torch.tensor([0],dtype=torch.long,device=p.device), p.shape[0]*p.shape[1]
        )
        normals.append(n[face_idx].clone()) #[nrenders*K,3]
    surface_normals=torch.stack(normals,0).view(bs,n_render,K,3)
    if visualize:
        for i in range(bs):
            for j in range(n_render):
                pdc = open3d.geometry.PointCloud()
                
                pdc.points = open3d.utility.Vector3dVector(p_3d[i,j].cpu().numpy())
                pdc.normals = open3d.utility.Vector3dVector(surface_normals[i,j].cpu().numpy())
                obj_mesh= open3d.geometry.TriangleMesh(
                    vertices=open3d.utility.Vector3dVector(verts[i].cpu().numpy()),
                    triangles=open3d.utility.Vector3iVector(faces[i].cpu().numpy()))
                mesh_wire=open3d.geometry.LineSet.create_from_triangle_mesh(obj_mesh)
                mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.2, origin=[0,0,0])
                open3d.visualization.draw_geometries([pdc,mesh_frame,mesh_wire])
    return surface_normals