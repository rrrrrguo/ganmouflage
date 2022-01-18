import pymeshlab
from pymeshlab import Mesh
import os
from joblib import delayed, Parallel
import numpy as np
from tqdm import tqdm
import open3d
import math
import glob


def process_one(path,target_dir):
    #os.makedirs(target, exist_ok=True)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    file_id=path.split('/')[-1].split('.')[0]
    ms.transform_translate_center_set_origin(traslmethod=1)
    max_dist = np.sqrt((ms.current_mesh().vertex_matrix()**2).sum(1).max(0))
    ms.matrix_set_from_translation_rotation_scale(scalex=0.5/max_dist,scaley=0.5/max_dist,scalez=0.5/max_dist)
    ms.simplification_clustering_decimation(threshold=0.015)
    verts=ms.current_mesh().vertex_matrix()
    faces=ms.current_mesh().face_matrix()
    verts[:,1]*=-1
    ms.clear()
    ms.add_mesh(Mesh(verts,faces))
    ms.save_current_mesh(f"{target_dir}/{file_id}.obj")



if __name__=="__main__":
    target_dir="../fake_animals_v4/"
    os.makedirs(target_dir,exist_ok=True)
    meshes=glob.glob("../animal_shape/*.ply")
    for mesh in meshes:
        process_one(mesh,target_dir)