# define all losses here
import tensorflow as tf
import numpy as np
from vggface import VGGFace

def landmark_loss(gt_landmarks, pred_landmarks):
    loss = tf.reduce_mean(tf.square(gt_landmarks, pred_landmarks), name='landmark_loss')
    return loss

def weighted_landmark_loss(gt_landmarks, pred_landmarks):
    lmk_weight = np.array([1.0] * 86, dtype=np.float32)

    # contour
    lmk_weight[1] = 2
    lmk_weight[4] = 2
    lmk_weight[8] = 2
    lmk_weight[12] = 2
    lmk_weight[15] = 2

    # eye
    lmk_weight[35] = 5
    lmk_weight[38] = 5
    lmk_weight[41] = 15
    lmk_weight[42] = 15

    lmk_weight[43] = 5
    lmk_weight[46] = 5
    lmk_weight[49] = 15
    lmk_weight[50] = 15

    # nose
    lmk_weight[59] = 5
    lmk_weight[60] = 5
    lmk_weight[63] = 5

    # mouth
    lmk_weight[66] = 5
    lmk_weight[70] = 5

    lmk_weight[68] = 10
    lmk_weight[71] = 10
    lmk_weight[79] = 10
    lmk_weight[82] = 10

    lmk_weight = np.reshape(lmk_weight, [1, 86, 1])

    loss = tf.reduce_mean(tf.square(gt_landmarks - pred_landmarks) * lmk_weight, name='landmark_loss')
    return loss

def photo_loss(images, renders, masks):
    EPS = 1e-8
    tmp = tf.reduce_sum(tf.square(images - renders) * masks, axis=-1, keepdims=True)
    tmp = tf.sqrt(tmp + EPS)
    sum_sq = tf.reduce_sum(tmp, axis=[1,2])
    sum_mask = tf.reduce_sum(masks, axis=[1,2])
    loss = tf.reduce_mean(tf.div(sum_sq, sum_mask+EPS), name='photo_loss')
    return loss


def gray_photo_loss(images, renders, masks):
    EPS = 1e-8
    gray_images = tf.reduce_mean(images, axis=-1, keepdims=True)
    gray_renders = tf.reduce_mean(renders, axis=-1, keepdims=True)
    sum_sq = tf.reduce_sum(tf.square(gray_images - gray_renders) * masks, axis=[1,2,3])
    sum_mask = tf.reduce_sum(masks, axis=[1,2])
    loss = tf.reduce_mean(tf.div(sum_sq, sum_mask), name='photo_loss')
    return loss



def id_loss(images, renders, vggpath):
    model = VGGFace(vggpath, False)

    inputs = tf.concat([images, renders], axis=0)
    layers, _, _ = model.encoder(inputs, False)
    z = layers['fc7']

    z_images, z_renders = tf.split(z, 2, axis=0)
    loss = tf.reduce_mean(tf.square(z_images - z_renders), name='id_loss')
    return loss

def reg_loss(para, para_name):
    loss = tf.reduce_mean(tf.square(para), name=para_name + '_loss')
    return loss

def smooth_tri_loss(tri_attrs, tri_pairs, thres, attr_name):
    tri1_ind, tri2_ind = tf.split(tri_pairs, 2, axis=-1)
    tri1_attrs = tf.gather(tri_attrs, tri1_ind, axis=1)
    tri2_attrs = tf.gather(tri_attrs, tri2_ind, axis=1)

    error = tf.maximum(tf.square(tri1_attrs - tri2_attrs), thres)
    loss = tf.reduce_mean(error, name=attr_name + '_smooth_tri_loss')
    return loss

def smooth_uv_loss(uv_attrs, attr_name):
    dx1 = uv_attrs[:,:,1:,:]
    dx2 = uv_attrs[:,:,:-1,:]
    dy1 = uv_attrs[:,1:,:,:]
    dy2 = uv_attrs[:,:-1,:,:]
    loss = tf.reduce_mean(tf.square(dx1 - dx2)) + \
            tf.reduce_mean(tf.square(dy1 - dy2))
    return loss

def smooth_slope_uv_loss(uv_attrs, attr_name):
    dx1 = uv_attrs[:,1:,1:,:]
    dx2 = uv_attrs[:,:-1,:-1,:]
    dy1 = uv_attrs[:,:-1,1:,:]
    dy2 = uv_attrs[:,1:,:-1,:]
    loss = tf.reduce_mean(tf.square(dx1 - dx2)) + \
            tf.reduce_mean(tf.square(dy1 - dy2))
    return loss


def smooth_duv_loss(uv_attrs, attr_name):
    dx1 = uv_attrs[:,:,1:,:]
    dx2 = uv_attrs[:,:,:-1,:]
    dx1_dx1 = dx1[:,:,1:,:]
    dx2_dx1 = dx1[:,:,:-1,:]
    dx1_dx2 = dx2[:,:,1:,:]
    dx2_dx2 = dx2[:,:,:-1,:]

    dy1 = uv_attrs[:,1:,:,:]
    dy2 = uv_attrs[:,:-1,:,:]
    dy1_dy1 = dy1[:,1:,:,:]
    dy2_dy1 = dy1[:,:-1,:,:]
    dy1_dy2 = dy2[:,1:,:,:]
    dy2_dy2 = dy2[:,:-1,:,:]
    loss = tf.reduce_mean(tf.square(dx1_dx1 - dx2_dx1)) + \
            tf.reduce_mean(tf.square(dx1_dx2 - dx2_dx2)) + \
            tf.reduce_mean(tf.square(dy1_dy1 - dy2_dy1)) + \
            tf.reduce_mean(tf.square(dy1_dy2 - dy2_dy2))
    return loss


def smooth_loss(ver_attrs, ver_neighbors, thres, attr_name):
    batch_size, n_ver, n_channels = ver_attrs.get_shape().as_list()
    n_ver_neighbor_pair = ver_neighbors.get_shape().as_list()[0]

    with tf.variable_scope(attr_name):

        var_sum_of_neighbor_attrs = tf.get_variable(
                'ver_sum_of_neighbor_attrs',
                [batch_size, n_ver, n_channels],
                tf.float32,
                tf.zeros_initializer(),
                trainable=False
                )

        var_sum_of_neighbor_counts = tf.get_variable(
                'ver_sum_of_counts',
                [batch_size, n_ver, 1],
                tf.float32,
                tf.zeros_initializer(),
                trainable=False
                )

    init_sum_of_neighbor_attrs = tf.zeros_like(var_sum_of_neighbor_attrs)
    init_sum_of_neighbor_counts = tf.zeros_like(var_sum_of_neighbor_counts)
    assign_op = tf.group([
        tf.assign(var_sum_of_neighbor_attrs, init_sum_of_neighbor_attrs),
        tf.assign(var_sum_of_neighbor_counts, init_sum_of_neighbor_counts)
        ], name='assign_op')

    with tf.control_dependencies([assign_op]):
        to_ver_ids, from_ver_ids = tf.split(ver_neighbors, 2, axis=1)
        tmp_ver_neighbor_attrs = tf.gather(ver_attrs, tf.squeeze(from_ver_ids), axis=1)
        tmp_ver_neighbor_counts = tf.ones([batch_size, n_ver_neighbor_pair, 1], tf.float32)

        batch_indices = tf.reshape(
                        tf.tile(tf.expand_dims(tf.range(batch_size),axis=1),[1,n_ver_neighbor_pair]),
                        [batch_size, n_ver_neighbor_pair, 1], name='batch_indices')
        to_ver_ids = tf.tile(tf.expand_dims(to_ver_ids, axis=0), [batch_size,1,1])
        batch_to_ver_ids = tf.concat([batch_indices, to_ver_ids], axis=2)
        var_sum_of_neighbor_attrs = tf.scatter_nd_add(var_sum_of_neighbor_attrs,
                batch_to_ver_ids, tmp_ver_neighbor_attrs)
        var_sum_of_neighbor_counts = tf.scatter_nd_add(var_sum_of_neighbor_counts,
                batch_to_ver_ids, tmp_ver_neighbor_counts)
        mean_neighbor_attrs = tf.div(var_sum_of_neighbor_attrs, var_sum_of_neighbor_counts + 1e-8)

        error = tf.maximum(tf.square(mean_neighbor_attrs - ver_attrs), thres)
        loss = tf.reduce_mean(error , name=attr_name + '_smooth_loss')
        return loss


def landmark2d_loss(gt_landmark, ver_xy, ver_mask, ver_normal, keypoint_indices):
    N = gt_landmark.get_shape().as_list()[1]
    MAX = 1e3
    pred_landmark = tf.gather(ver_xy, keypoint_indices, axis=1)
    norm_landmark = tf.gather(ver_normal, keypoint_indices, axis=1)
    normal_z = tf.reshape(tf.split(norm_landmark, 3, axis=2)[-1], [-1, N])

    standard_landmark_losses = tf.reduce_mean(
        tf.square(gt_landmark - pred_landmark), axis=2
    )

    invisible_losses = []
    lmk_weight = np.array([1.0] * 18,dtype=np.float32)
    lmk_weight[1] = 5
    lmk_weight[4] = 5
    lmk_weight[8] = 5
    lmk_weight[12] = 5
    lmk_weight[15] = 5

    for i in range(N):  # each ground truth, find the nearest vertex value
        gt_lmk = gt_landmark[:, i : (i + 1), :]

        loss = tf.reduce_sum(
            tf.square(gt_lmk - ver_xy) * ver_mask + MAX * (1 - ver_mask), axis=-1
        )
        loss = tf.reduce_min(loss, axis=1) * lmk_weight[i]
        loss = tf.where(
            tf.greater(loss, float(7.0 / 224 * 300)),
            float(7.0 / 224 * 300) * tf.ones_like(loss),
            loss,
        )
        invisible_losses.append(loss)

    invisible_losses = tf.reshape(tf.stack(invisible_losses, axis=0), [-1, 18])

    losses = tf.where(
        tf.greater(normal_z, 0.0), standard_landmark_losses, invisible_losses
    )
    loss = tf.reduce_mean(losses, name="landmark2d_loss")
    return loss


def disp_loss(ver_attrs, ver_masks, attr_name):
    tmp = tf.clip_by_value(tf.square(ver_attrs), 0, 10)
    sum_sq = tf.reduce_sum(tmp * ver_masks, axis=[1,2])
    sum_mask = tf.reduce_sum(ver_masks, axis=[1,2])
    loss = tf.reduce_mean(tf.div(sum_sq, sum_mask), name=attr_name + '_disp_loss')
    return loss

