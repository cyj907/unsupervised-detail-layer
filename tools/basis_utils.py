""" basis related operations """
import tensorflow as tf
import glob
import skimage.io
import numpy as np
import scipy.io
import imageio
import cv2
import json
import os


def get_geometry(basis3dmm, para_shape, para_exp):
    """ compute the geometry according to the 3DMM parameters.
    para_shape: shape parameter (199d)
    para_exp: expression parameter (29d)
    """
    shape_inc = tf.matmul(para_shape, basis3dmm['bases_shape'])
    exp_inc   = tf.matmul(para_exp, basis3dmm['bases_exp'])
    geo = basis3dmm['mu_shape'] + basis3dmm['mu_exp'] + shape_inc + exp_inc
    n_vtx = basis3dmm['bases_shape'].shape[1] // 3
    return tf.reshape(geo,[-1,n_vtx,3])


def get_texture(basis3dmm, para_tex):
    """ compute the texture according to texture parameter.
    para_tex: texture parameter (199d)
    """
    tex_inc = tf.matmul(para_tex,basis3dmm['bases_tex'])
    texture = tf.clip_by_value(basis3dmm['mu_tex'] + tex_inc,0,255)
    n_vtx = basis3dmm['bases_shape'].shape[1] // 3
    return tf.reshape(texture, [-1,n_vtx,3])


def add_z_to_UV(basis3dmm):
    uv = basis3dmm['ver_uv_ind']
    # add random values to prevent vertices from collapsing into the same pixels
    uv = uv + np.random.uniform(size=[uv.shape[0],2],low=0.,high=0.00001)
    uv = uv.astype(np.float32)
    z = np.reshape(basis3dmm['mu_shape'], [-1,3])[:,2:3]
    norm_z = (z - np.amin(z)) / (np.amax(z) - np.amin(z))
    uvz = np.concatenate([uv,-norm_z], axis=1)
    return uvz


def load_3dmm_basis(basis_path,
        ver_uv_ind_path,
        uv_face_mask_path,
        ver_wo_eyebrow_mask_path=None,
        ver_wo_nose_mask_path=None):
    basis3dmm = scipy.io.loadmat(basis_path)
    basis3dmm['bases_shape'] = np.transpose(basis3dmm['bases_shape'] * basis3dmm['sigma_shape']).astype(np.float32)
    basis3dmm['bases_exp'] = np.transpose(basis3dmm['bases_exp'] * basis3dmm['sigma_exp']).astype(np.float32)
    basis3dmm['bases_tex'] = np.transpose(basis3dmm['bases_tex'] * basis3dmm['sigma_tex']).astype(np.float32)
    basis3dmm['mu_shape'] = np.transpose(basis3dmm['mu_shape']).astype(np.float32)
    basis3dmm['mu_exp'] = np.transpose(basis3dmm['mu_exp']).astype(np.float32)
    basis3dmm['mu_tex'] = np.transpose(basis3dmm['mu_tex']).astype(np.float32)
    basis3dmm['tri'] = basis3dmm['tri'].astype(np.int32)

    # get vertex triangle relation ship
    vertex_tri_set = set()
    vertex_vertex_set = set()
    for i, tri in enumerate(basis3dmm['tri']):
        v1 = tri[0]
        v2 = tri[1]
        v3 = tri[2]
        vertex_tri_set.add((v1,i))
        vertex_tri_set.add((v2,i))
        vertex_tri_set.add((v1,i))
        vertex_tri_set.add((v3,i))
        vertex_tri_set.add((v2,i))
        vertex_tri_set.add((v3,i))

        vertex_vertex_set.add((v1, v2))
        vertex_vertex_set.add((v2, v1))
        vertex_vertex_set.add((v1, v3))
        vertex_vertex_set.add((v3, v1))
        vertex_vertex_set.add((v2, v3))
        vertex_vertex_set.add((v3, v2))
    vertex_tri_set = np.array(list(vertex_tri_set), np.int32)
    vertex_vertex_set = np.array(list(vertex_vertex_set), np.int32)
    basis3dmm['ver_tri'] = vertex_tri_set
    basis3dmm['ver_adj'] = vertex_vertex_set

    # crop bases
    basis3dmm['bases_shape'] = basis3dmm['bases_shape'][:80,:]
    basis3dmm['bases_tex'] = basis3dmm['bases_tex'][:80,:]

    ver_uv_ind = np.load(ver_uv_ind_path)['uv_ind']
    basis3dmm['ver_uv_ind'] = (ver_uv_ind / 512.0).astype(np.float32)

    uv_face_mask = cv2.imread(uv_face_mask_path)
    if len(uv_face_mask.shape) == 3:
        uv_face_mask = uv_face_mask[:,:,0] / 255.0
    uv_face_mask = uv_face_mask.astype(np.float32)
    basis3dmm['uv_face_mask'] = uv_face_mask

    # get the neighboring relationship of triangles
    edge_to_triangles = {}
    for idx, tri in enumerate(basis3dmm['tri']):
        v1 = tri[0]
        v2 = tri[1]
        v3 = tri[2]
        try:
            edge_to_triangles[(v1,v2)].append(idx)
        except Exception:
            edge_to_triangles[(v1,v2)] = [idx]

        try:
            edge_to_triangles[(v2,v1)].append(idx)
        except Exception:
            edge_to_triangles[(v2,v1)] = [idx]

        try:
            edge_to_triangles[(v1,v3)].append(idx)
        except Exception:
            edge_to_triangles[(v1,v3)] = [idx]

        try:
            edge_to_triangles[(v3,v1)].append(idx)
        except Exception:
            edge_to_triangles[(v3,v1)] = [idx]

        try:
            edge_to_triangles[(v2,v3)].append(idx)
        except Exception:
            edge_to_triangles[(v2,v3)] = [idx]

        try:
            edge_to_triangles[(v3,v2)].append(idx)
        except Exception:
            edge_to_triangles[(v3,v2)] = [idx]

    tri_pairs = []
    for key in edge_to_triangles:
        relations = edge_to_triangles[key]
        for item_a in relations:
            for item_b in relations:
                if item_a < item_b:
                    tri_pairs.append((item_a, item_b))
    tri_pairs = set(tri_pairs)
    tri_pairs = np.array(list(tri_pairs), np.int32)
    basis3dmm['tri_pairs'] = tri_pairs

    # keypoints
    kpts_86 = [22502,22653,22668,22815,22848,44049,46010,47266,47847,48436,49593,51577,31491,31717,32084,32197,32175,38779,39392,39840,40042,40208,39465,39787,39993,40213,40892,41087,41360,41842,42497,40898,41152,41431,13529,1959,3888,5567,6469,5450,3643,4920,4547,9959,10968,12643,14196,12785,11367,11610,12012,8269,8288,8302,8192,6837,9478,6499,10238,6002,10631,6755,7363,8323,9163,9639,5652,7614,8216,8935,11054,8235,6153,10268,9798,6674,6420,10535,7148,8227,9666,6906,8110,9546,7517,8837]
    kpts_86 = np.array(kpts_86, np.int32) - 1
    basis3dmm['keypoints'] = kpts_86

    if ver_wo_eyebrow_mask_path is not None:
        ver_wo_eyebrow_mask = np.load(ver_wo_eyebrow_mask_path)
        ver_wo_eyebrow_mask = ver_wo_eyebrow_mask.astype(np.float32)
        ver_wo_eyebrow_mask = np.reshape(ver_wo_eyebrow_mask, [-1, 1])

        for i in range(10):
            ver_wo_eyebrow_mask = laplace_smoothing(ver_wo_eyebrow_mask, basis3dmm['ver_adj'])
        ver_wo_eyebrow_mask = np.squeeze(ver_wo_eyebrow_mask)
        basis3dmm['mask_wo_eyebrow'] = ver_wo_eyebrow_mask[np.newaxis,...]

    if ver_wo_nose_mask_path is not None:
        ver_wo_nose_mask = np.load(ver_wo_nose_mask_path)
        ver_wo_nose_mask = ver_wo_nose_mask.astype(np.float32)
        ver_wo_nose_mask = np.reshape(ver_wo_nose_mask, [-1, 1])

        for i in range(10):
            ver_wo_nose_mask = laplace_smoothing(ver_wo_nose_mask, basis3dmm['ver_adj'])
        ver_wo_nose_mask = np.squeeze(ver_wo_nose_mask)
        basis3dmm['mask_wo_nose'] = ver_wo_nose_mask[np.newaxis,...]

    return basis3dmm



def laplace_smoothing(ver, ver_adj):
    # ver: numpy array, [N, x]
    # ver_adj: [M, 2], (ver_id, ver_adj_id), vertex index starting from 0
    N = np.amax(ver_adj)
    assert(len(ver.shape) == 2)
    ver_attrs = np.zeros_like(ver)
    ver_counts = np.zeros([ver.shape[0], 1], np.float32)
    for (v_id, v_adj_id) in ver_adj:
        ver_attrs[v_id,:] = ver_attrs[v_id,:] + ver[v_adj_id,:]
        ver_counts[v_id,:] += 1
    ver_attrs = ver_attrs / (ver_counts + 1e-8)
    return ver_attrs



