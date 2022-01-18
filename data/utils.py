import numpy as np
from scipy.io import loadmat
import cv2
import torch
import os
from PIL import Image
from sklearn.metrics import pairwise_distances
import random

def read_bundler(fname):
    f = open(fname)
    f.readline()
    ncams = int(f.readline().split()[0])
    Rt = []
    focals = []
    for i in range(ncams):
        focals.append(float(f.readline().split()[0]))
        R = np.array(
            [list(map(float, f.readline().split())) for x in range(3)])
        t = np.array(list(map(float, f.readline().split())))
        Rt.append((R, t))
    f.close()
    return focals, Rt

def calculate_rotation(cube_vertices,save_type=0):
    cube_center=cube_vertices.mean(0)
    no_translate_coord=cube_vertices-cube_center.reshape(-1,3)
    cube_diagonal=pairwise_distances(no_translate_coord).max()
    if save_type==0:
        fake_cube_vertices=np.array([[-0.5, -0.5, -0.5],
                                    [-0.5,-0.5, 0.5],
                                    [0.5, -0.5, -0.5],
                                    [0.5, -0.5, 0.5],
                                    [-0.5, 0.5, -0.5],
                                    [-0.5, 0.5, 0.5],
                                    [0.5, 0.5, -0.5],
                                    [0.5, 0.5, 0.5],
                                    ])/np.sqrt(3)*cube_diagonal
    elif save_type==1:
        fake_cube_vertices=np.array([[-0.5, 0.5, -0.5],
                                    [-0.5,0.5, 0.5],
                                    [0.5, 0.5, -0.5],
                                    [0.5, 0.5, 0.5],
                                    [-0.5, -0.5, -0.5],
                                    [-0.5, -0.5, 0.5],
                                    [0.5, -0.5, -0.5],
                                    [0.5, -0.5, 0.5],
                                    ])/np.sqrt(3)*cube_diagonal
    else:
        fake_cube_vertices=np.array([[-0.5, -0.5, -0.5],
                                    [-0.5,-0.5, 0.5],
                                    [-0.5, 0.5, -0.5],
                                    [-0.5, 0.5, 0.5],
                                    [0.5, -0.5, -0.5],
                                    [0.5, -0.5, 0.5],
                                    [0.5, 0.5, -0.5],
                                    [0.5, 0.5, 0.5],
                                    ])/np.sqrt(3)*cube_diagonal
    C=no_translate_coord.T @ fake_cube_vertices #[3,3]
    U,_,Vh=np.linalg.svd(C)
    cube_R= U@Vh
    #print(np.abs((cube_R @ (fake_cube_vertices.T)).T-no_translate_coord).sum())
    return cube_R


def get_scene_parameters_olddata(scene_dir,cube_save_type=0,cube_name=None):
    focals, Rts = read_bundler(scene_dir + "bundle/bundle.out")
    # filter our good camera
    view_filenames=[f"view{x+1}.jpg" for x in range(len(focals))]
    if os.path.isfile(scene_dir + "good_cams.txt"):
        with open(scene_dir + "good_cams.txt") as f:
            good_cameras = f.readlines()
        good_cameras = [int(x[:-1]) for x in good_cameras]
    else:
        good_cameras = list(range(1,1+len(focals)))
    #select good views from good_cams.txt
    focals = [focals[x - 1] for x in good_cameras]
    Rts = [Rts[x - 1] for x in good_cameras]
    view_filenames = [view_filenames[x-1] for x in good_cameras]

    #assert len(good_cameras) >= 2, "At least 2 good views is needed"
    # load cube
    cube_file = scene_dir + 'cube.mat' if not cube_name else cube_name
    box_vertices = load_cube_vertices(cube_file)
    #calculate cube rotation R from coordinates
    cube_R=calculate_rotation(box_vertices,save_type=cube_save_type)

    return focals, Rts, view_filenames, box_vertices,cube_R


def load_cube_vertices(fname):
    m = loadmat(fname)
    if np.ndim(m['world_pos']) == 2 and m['world_pos'].shape[1] == 3:
        face_pts = np.array(m['world_pos'], 'd')
    else:
        face_pts = np.array([m['world_pos'].squeeze()[i].flatten() for i in range(len(m['world_pos'].squeeze()))], 'd')
    assert face_pts.shape==(8,3), face_pts.shape
    return face_pts


def plane_estimation(vertices):
    '''
    Parameters
    ----------
    vertices 4x3 matrix containing 4 lower vertices

    Returns
    -------
    plane  [a,b,c] ax+by+cz=1
    '''
    return np.linalg.pinv(vertices) @ np.ones(vertices.shape[0])


def sample_position(plane, reference, distance):
    '''
    Parameters
    ----------
    plane        [3]([a,b,c]) ax+by+cz+d=0
    reference    reference point coordinate [3], center of the cube
    distance     max distance to the reference point

    Returns
    -------
    t   translation to the new place
    '''
    # n=np.array([plane[0],plane[1],plane[2]])
    v1 = np.array([0, -plane[2], plane[1]])
    v2 = np.array([1, -plane[0] * plane[1] / (plane[1] ** 2 + plane[2] ** 2),
                   -plane[0] * plane[2] / (plane[1] ** 2 + plane[2] ** 2)])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    rho = np.random.uniform(0, distance)
    theta = np.random.uniform(0, 2 * np.pi)
    alpha1 = rho * np.cos(theta)
    alpha2 = rho * np.sin(theta)
    return reference + alpha1 * v1 + alpha2 * v2


def get_sample_boundary(plane, reference, distance,points=100):
    '''
    Parameters
    ----------
    plane        [3]([a,b,c]) ax+by+cz+d=0
    reference    reference point coordinate [3], center of the cube
    distance     max distance to the reference point

    Returns
    -------
    t   translation to the new place
    '''
    # n=np.array([plane[0],plane[1],plane[2]])
    v1 = np.array([0, -plane[2], plane[1]])
    v2 = np.array([1, -plane[0] * plane[1] / (plane[1] ** 2 + plane[2] ** 2),
                   -plane[0] * plane[2] / (plane[1] ** 2 + plane[2] ** 2)])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    #theta = np.random.uniform(0, 2 * np.pi)
    #alpha1 = rho * np.cos(theta)
    #alpha2 = rho * np.sin(theta)
    thetas=np.linspace(0,np.pi*2,num=points) 
    alpha1=distance*np.cos(thetas).reshape(points,1)
    alpha2=distance*np.sin(thetas).reshape(points,1)
    return reference.reshape(1,3) + alpha1 * v1.reshape(1,3) + alpha2 * v2.reshape(1,3)

def project_to_image(P, vertices):
    '''
    Parameters
    ----------
    get_P         3x4 matrix K@[R|t]
    vertices  [N,3] vertices coordinates in world

    Returns
    -------
    [N,2] in screen space
    '''
    X = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1).T  # [4,N]
    homog = P @ X  # [3,N]
    homog = homog[:-1] / homog[-1].reshape(1, -1)
    return homog.T  # [N,2]


def load_background_image(path, shape):
    '''
        Load background image and resize, scale [0,1], return array [H,W,3]
    '''
    background = cv2.imread(path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    background = cv2.resize(background, shape,interpolation=cv2.INTER_LANCZOS4)
    background = background.astype(np.float32) / 255
    # background=background.transpose(2,0,1)
    return background

def load_background_image_aa(path, shape):
    '''
        Load background image and resize, scale [0,1], return array [H,W,3]
    '''
    background = Image.open(path)
    background = background.resize(shape,Image.ANTIALIAS)
    #background = cv2.resize(background, shape,interpolation=cv2.INTER_LANCZOS4)
    background = np.asarray(background)
    background = background.astype(np.float32) / 255
    # background=background.transpose(2,0,1)
    return background


def cameraToNDC(focal_screen, px_screen, py_screen, image_width, image_height):
    '''
        Convert camera parameters to NDC space
    '''
    fx = focal_screen * 2.0 / image_width
    fy = focal_screen * 2.0 / image_height
    px = - (px_screen - image_width / 2.0) * 2.0 / image_width
    py = - (py_screen - image_height / 2.0) * 2.0 / image_height
    return fx, fy, px, py


def collate_fn(data, to_list_key=['verts', 'faces', 'obj_id','image_meta']):
    result = dict()
    for k in data[0]:
        if k in to_list_key:
            result[k] = [x[k] for x in data]
        else:
            result[k] = torch.stack([torch.tensor(x[k]) for x in data], 0)
    return result


def draw_gaussian_mask(shape, center, obj_size):
    sigma = np.mean(obj_size) / 4
    x_mesh, y_mesh = np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0]))
    mask = np.exp(-((x_mesh - center[0]) ** 2 + (y_mesh - center[1]) ** 2) / (2 * sigma ** 2))
    return mask
