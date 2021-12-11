""" utility functions used in multiple scripts """
import cv2
import numpy as np
import skimage.io
import tensorflow as tf
import os
import random
from absl import logging
from PIL import Image
from tools.const import *

class MaskCreator(object):
    """ use for training only. """

    @staticmethod
    def create_coarse_photo_loss_mask_from_seg(seg):
        # create photo loss mask from face segmentation
        # use skin, nose, eyeglass, lrow, rbrow, ulip, llip
        mask = (
            seg[:, :, :, SEG_SKIN]
            + seg[:, :, :, SEG_NOSE]
            + seg[:, :, :, SEG_LBROW]
            + seg[:, :, :, SEG_RBROW]
            + seg[:, :, :, SEG_ULIP]
            + seg[:, :, :, SEG_LLIP]
        )
        mask = tf.expand_dims(mask, axis=-1)
        return mask


    @staticmethod
    def erosion(mask, k):
        neg_mask = 1 - mask
        kernel = tf.ones([k,k,1,1], tf.float32) / (k*k)
        neg_mask = tf.nn.conv2d(neg_mask, kernel, [1,1,1,1], padding='SAME')
        mask = 1 - neg_mask
        return mask

    @staticmethod
    def create_fine_photo_loss_mask_from_seg(seg):
        # create photo loss mask from face segmentation
        # use skin, nose, eyeglass, lrow, rbrow, ulip, llip
        mask = seg[:, :, :, SEG_SKIN]
        mask = tf.expand_dims(mask, axis=-1)
        return mask


def detect_glassframe(rgb_img, seg_img):
    # apply edge detection filter
    # use filter to assist expansion valid region
    # input range [0, 1]
    dX = seg_img[:, 1:] - seg_img[:, :-1]
    dX = np.pad(dX, ((0, 0), (0, 1)), "constant")
    dY = seg_img[1:, :] - seg_img[:-1, :]
    dY = np.pad(dY, ((0, 1), (0, 0)), "constant")
    G = np.sqrt(np.square(dX) + np.square(dY))
    G[G > 0.1] = 1
    k = 10
    kernel = np.ones((k, k), np.float32) / (k * k)
    mask = cv2.filter2D(G, -1, kernel)
    mask[mask > 0.01] = 1

    # convert rgb to hsv and use v to threshold a binary mask
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    v_img = hsv_img[:, :, 2]
    v_mask = (v_img < 0.6).astype(np.float32)

    # logical and (filter_valid, v_mask, seg)
    glassframe = v_mask * mask * seg_img
    return glassframe


def tf_detect_glassframe(rgb_img, seg_img, G_list):
    # input range [0,1]
    dX = seg_img[:, :, 1:] - seg_img[:, :, :-1]
    dX = tf.pad(dX, ((0, 0), (0, 0), (0, 1)), "constant")
    dY = seg_img[:, 1:, :] - seg_img[:, :-1, :]
    dY = tf.pad(dY, ((0, 0), (0, 1), (0, 0)), "constant")
    G = tf.sqrt(tf.square(dX) + tf.square(dY))
    G = tf.where(tf.greater(G, 0.1), tf.ones_like(G), tf.zeros_like(G))
    G = tf.expand_dims(G, axis=3)

    k = 10
    kernel = np.ones((k, k), np.float32) / (k * k)
    kernel = tf.reshape(kernel, [k, k, 1, 1])

    G_list = tf.split(G, G_list.get_shape().as_list()[-1], axis=-1)
    G_out_list = []
    for g_channel in G_list:
        g_out_channel = tf.conv2d(g_channel, kernel, strides=[1,1,1,1], padding='SAME')
        G_out_list.append(g_out_channel)
    mask = tf.concat(G_out_list, axis=-1)
    mask = tf.where(tf.greater(mask, 0.01), tf.ones_like(mask), tf.zeros_like(mask))
    mask = tf.squeeze(mask)

    # convert rgb to hsv
    hsv_img = tf_rgb_to_hsv(rgb_img)
    v_img = hsv_img[:, :, :, 2]
    v_mask = tf.where(tf.less(v_img, 0.6), tf.ones_like(v_img), tf.zeros_like(v_img))
    glassframe = v_mask * mask * seg_img
    return glassframe



