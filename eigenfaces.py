import glob
import json
import pickle

import cv2
import numpy as np
import PIL
from PIL import Image


def read_imgs(dir):
    """convert imgs into a matrix
    return:(m*n)*p matrix
    p: num of imgs
    """
    img_dir_list = glob.glob(dir + '\*.tga')
    imgs = []
    for img_dir in img_dir_list:
        img_hist = img_processor(img_dir)
        img_hist = img_hist.flatten()

        imgs.append(img_hist)

    imgs = np.array(imgs).T

    return imgs


def img_processor(img_dir):
    """input dir
    return ndarray
    """
    img = Image.open(img_dir).convert('L')
    img = img.resize((50, 50))
    img = np.array(img)
    img_hist = cv2.equalizeHist(img)
    return img_hist

def img_processor2(img):
    """input dir
    return ndarray
    """
    img = img.resize((50, 50))
    img = np.array(img)
    img_hist = cv2.equalizeHist(img)
    return img_hist

def pca(imgs):
    M = np.dot(imgs.T, imgs)
    e, ev = np.linalg.eigh(M)
    eigenfaces = np.dot(imgs, ev)

    return eigenfaces


def save_faces(faces, mean, dir):
    num_faces = faces.shape[1]
    for i in range(num_faces):
        face = faces[:, i:i + 1]
        face = face + mean

        face_T = face.T
        face_T.resize((50, 50))
        cv2.imwrite(dir + str(i) + '.jpg', face_T)


# single image test.
def reconstruct(dir, project_mat, mean):
    src = img_processor(dir)
    src.resize((2500, 1))
    src = src - mean

    vec_in_space = np.dot(project_mat, src)
    components = vec_in_space * project_mat
    save_faces(components.T, mean, 'components/')
    recon_vec = np.dot(vec_in_space.T, project_mat).T
    save_faces(recon_vec, mean, '')

    err = (src-recon_vec)/255
    err = np.sum(np.square(err))/2500
    err = np.sqrt(err)

    return err



def reconstruct2(src, project_mat, mean):
    src.resize((2500, 1))
    src = src - mean

    vec_in_space = np.dot(project_mat, src)
    pickle.dump(vec_in_space,fp=open('faces_in_pca.json','w'))
    components = vec_in_space * project_mat
    save_faces(components.T, mean, 'components/')
    recon_vec = np.dot(vec_in_space.T, project_mat).T
    save_faces(recon_vec, mean, '')

    err = (src-recon_vec)/255
    err = np.sum(np.square(err))/2500
    err = np.sqrt(err)

    return err
# ---------------------------------------------------------------
def get_save_eigenfaces(train_dir='faces/smiling_cropped'):
    imgs = read_imgs(train_dir)
    dim_imgs = imgs.shape[0]
    mean = imgs.mean(axis=1)
    mean.resize((dim_imgs, 1))
    imgs = imgs - mean

    eigenfaces = pca(imgs)
    # reverse
    eigenfaces = eigenfaces[:, ::-1]
    eigenfaces = eigenfaces[:, :2]
    project_mat = eigenfaces.T
    save_faces(eigenfaces, mean, 'res/')

    return project_mat, mean, imgs


def get_save_recon_face(project_mat, mean):
    # unit
    for i in range(project_mat.shape[0]):
        vec = project_mat[i]
        norm = np.sum(np.square(vec))
        norm = np.sqrt(norm)
        vec = vec / norm
        project_mat[i] = vec

    # eigenfaces vary with project_mat
    reconstruct('faces/smiling_cropped/05.tga', project_mat, mean)


project_mat, mean, _ = get_save_eigenfaces()
get_save_recon_face(project_mat, mean)

