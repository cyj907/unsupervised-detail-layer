import numpy as np
import cv2
import os

def process_image_landmark(img, landmark, size = 300, orig=False):
    # use landmarks to crop face
    min_x, min_y = np.amin(landmark, 0)
    max_x, max_y = np.amax(landmark, 0)

    c_x = min_x + 0.5 * (max_x - min_x)
    c_y = min_y + 0.4 * (max_y - min_y)
    half_x = (max_x - min_x) * 0.8
    half_y = (max_y - min_y) * 0.6
    half_width = max(half_x, half_y)

    st_x = c_x - half_width
    en_x = c_x + half_width
    st_y = c_y - half_width
    en_y = c_y + half_width

    pad_t = int(max(0 - st_y, 0) + 0.5)
    pad_b = int(max(en_y - img.shape[0], 0) + 0.5)
    pad_l = int(max(0 - st_x, 0) + 0.5)
    pad_r = int(max(en_x - img.shape[1], 0) + 0.5)

    st_x_ = int(st_x + 0.5)
    en_x_ = int(en_x + 0.5)
    st_y_ = int(st_y + 0.5)
    en_y_ = int(en_y + 0.5)

    # pad top and bottom
    pad_img = np.concatenate([np.zeros((pad_t, img.shape[1], 3)),img,np.zeros((pad_b, img.shape[1], 3))],axis=0)
    # pad left and right
    pad_img = np.concatenate([np.zeros((pad_img.shape[0],pad_l,3)),pad_img,np.zeros((pad_img.shape[0],pad_r,3))],axis=1)
    crop_img = pad_img[(st_y_+pad_t):(en_y_+pad_t),(st_x_+pad_l):(en_x_+pad_l),:]

    start_landmark = np.array([[st_x, st_y]])
    crop_landmark = landmark - start_landmark

    lmk_x = landmark[:,0]
    lmk_x = (lmk_x - st_x) / crop_img.shape[1] * size
    lmk_y = landmark[:,1]
    lmk_y = (lmk_y - st_y) / crop_img.shape[0] * size
    landmark = np.stack([lmk_x + 0.5,lmk_y + 0.5],axis=1)

    if crop_img.shape[0] != crop_img.shape[1]:
        crop_img = cv2.resize(crop_img, (crop_img.shape[0], crop_img.shape[0]))

    old_crop_img = crop_img.copy()
    cur_h, cur_w, _ = crop_img.shape
    while cur_h // 4 > size:
        crop_img = cv2.GaussianBlur(crop_img, (3,3), 1)
        crop_img = crop_img[::2,:,:]
        crop_img = crop_img[:,::2,:]
        cur_h, cur_w, _ = crop_img.shape
    crop_img = cv2.resize(crop_img, (size, size))

    ## crop reference image
    if orig:
        return crop_img, landmark, old_crop_img, crop_landmark
    else:
        return crop_img, landmark

