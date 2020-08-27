import glob
import os
import scipy.io
import skimage.io
import numpy as np
import cv2

MIN_ABS_Z = 1e-12

# 512 UV map
ALPHA1 = 160
BETA1  = 256
ALPHA2 = -2.2
BETA2  = 220
W = 512

# 256 uv map
#ALPHA1 = 80
#BETA1  = 128
#ALPHA2 = -1.15
#BETA2  = 110
#W = 256

def get_uv_ind():
    # load mat
    mat = scipy.io.loadmat('../resources/BFM2009_Model.mat')
    mu_shape = np.reshape(mat['mu_shape'],[-1,3])
    mu_exp = np.reshape(mat['mu_exp'],[-1,3])
    mu_geo = mu_shape + mu_exp

    x, y, z = np.split(mu_geo,3,1)
    v = ALPHA1 * np.arctan(x/(np.abs(z)+MIN_ABS_Z)*np.sign(z)) + BETA1
    u = ALPHA2 * y + BETA2

    v = v.astype(np.int32)
    u = u.astype(np.int32)
    uv_ind = np.concatenate([u, v], axis=1)
    print(uv_ind.shape)

    np.savez('vertex_uv_ind_%d.npz' % W, uv_ind=uv_ind)


def uv_map_one(geometry,texture):
    x, y, z = np.split(geometry,3,1)
    v = ALPHA1 * np.arctan(x/(np.abs(z)+MIN_ABS_Z)*np.sign(z)) + BETA1
    u = ALPHA2 * y + BETA2

    canvas = np.zeros((W,W,3))
    z_buffer = np.ones((W,W)) * np.amin(z)
    for i, (coord_x, coord_y) in enumerate(zip(u,v)):
        coord_x = int(coord_x + 0.5)
        coord_y = int(coord_y + 0.5)

        #coord_x = np.clip(coord_x,0,W - 1)
        #coord_y = np.clip(coord_y,0,W - 1)

        if z[i] > z_buffer[coord_x, coord_y]:
            canvas[coord_x,coord_y] = texture[i]
            z_buffer[coord_x, coord_y] = z[i]
    canvas[canvas < 0] = 0
    canvas[canvas > 255] = 255
    return canvas.astype(np.uint8)


def mean_face_uv_mapping():
    # load mat
    mat = scipy.io.loadmat('../resources/BFM2009_Model.mat')
    mu_shape = np.reshape(mat['mu_shape'],[-1,3])
    mu_exp = np.reshape(mat['mu_exp'],[-1,3])
    mu_geo = mu_shape + mu_exp
    mu_tex = np.reshape(mat['mu_tex'],[-1,3])
    uv_map = uv_map_one(mu_geo,mu_tex)
    skimage.io.imsave('tmp.png', uv_map)


if __name__ == '__main__':
    mean_face_uv_mapping()
    get_uv_ind()

