"""
Define a new dataset.

2020/1/22: include tf records
"""

import numpy as np
import tensorflow as tf
import os
import cv2
from tools.const import flip_vtx_map68, flip_vtx_map86
from tools.misc import detect_glassframe


def py_resize(image, shape):
    shape = (shape[0], shape[1])
    image = cv2.resize(image, shape)
    return image

class DataSet(object):

    def __init__(self):
        """
        This dataset only contains image, landmark2d, landmark3d, and segmentation.

        image: [300,300,3]
        landmark2d: [68,2]
        landmark3d: [86,2]
        segmentation: [300,300] => [300,300,19]
        """
        pass

    @staticmethod
    def create_one_hot_seg_label(seg_tensor):
        seg_labels = []
        for i in range(19):
            seg = tf.cast(tf.equal(seg_tensor, i), tf.float32)
            seg_labels.append(seg)
        seg_labels = tf.stack(seg_labels, axis=-1)
        return seg_labels

    @staticmethod
    def augment(image, landmark2d, landmark3d, segmentation, glassframe, \
            is_color=True, is_scale=True, is_flip=True):

        # random shift
        if is_color:
            image = DataSet.random_color_shift(image)

        # random flip
        if is_flip:
            image, landmark2d, landmark3d, segmentation, glassframe = \
                    DataSet.random_flip(image, landmark2d, landmark3d, segmentation, glassframe)
        # random scale
        if is_scale:
            image, landmark2d, landmark3d, segmentation, glassframe = \
                    DataSet.random_resize_and_crop(image, landmark2d, landmark3d, segmentation, glassframe)

        return image, landmark2d, landmark3d, segmentation, glassframe

    @staticmethod
    def random_flip(image, landmark2d, landmark3d, segmentation, glassframe):
        rand_num = np.random.sample()
        if rand_num > 0.5:
            flip_image = np.flip(image, axis=1)

            flip_landmark2d = []
            for i in range(landmark2d.shape[0]):
                flip_landmark2d.append(landmark2d[flip_vtx_map68[i]])
            flip_landmark2d = np.array(flip_landmark2d)
            flip_landmark2d[:,0] = image.shape[0] - flip_landmark2d[:,0]

            flip_landmark3d = []
            for i in range(landmark3d.shape[0]):
                flip_landmark3d.append(landmark3d[flip_vtx_map86[i]])
            flip_landmark3d = np.array(flip_landmark3d)
            flip_landmark3d[:,0] = image.shape[0] - flip_landmark3d[:,0]

            flip_segmentation = np.flip(segmentation, axis=1)
            flip_glassframe = np.flip(glassframe, axis=1)

            image = flip_image
            landmark2d = flip_landmark2d
            landmark3d = flip_landmark3d
            segmentation = flip_segmentation
            glassframe = flip_glassframe
        return image, landmark2d, landmark3d, segmentation, glassframe

    @staticmethod
    def random_resize_and_crop(image, landmark2d, landmark3d, segmentation, glassframe):
        import cv2
        # randomly resize image to 0.8 - 1.2x and crop
        scale = np.random.uniform(low=0.9,high=1.1)
        H, W = image.shape[:2]
        resize_image = cv2.resize(image, (int(H*scale),int(W*scale)))
        resize_landmark2d = landmark2d * scale
        resize_landmark3d = landmark3d * scale
        resize_seg_list = []
        for i in range(segmentation.shape[-1]):
            seg = segmentation[:,:,i]
            resize_seg = cv2.resize(seg, (int(H*scale),int(W*scale)), interpolation=cv2.INTER_NEAREST)
            resize_seg_list.append(resize_seg)
        resize_seg = np.stack(resize_seg_list, axis=2)

        resize_glassframe = cv2.resize(glassframe, (int(H*scale),int(W*scale)), interpolation=cv2.INTER_NEAREST)

        # pad or crop according to situation
        if scale < 1.0:
            pad_size_a = int((H - H*scale) * 0.5)
            pad_size_b = H - int(H*scale) - pad_size_a
            out_image = np.pad(resize_image,
                    ((pad_size_a,pad_size_b),(pad_size_a,pad_size_b),(0,0)),'constant')
            out_landmark2d = resize_landmark2d + np.array([pad_size_a,pad_size_a])
            out_landmark3d = resize_landmark3d + np.array([pad_size_a,pad_size_a])
            out_seg = np.pad(resize_seg,
                    ((pad_size_a,pad_size_b),(pad_size_a,pad_size_b),(0,0)),'constant')
            out_glassframe = np.pad(resize_glassframe,
                    ((pad_size_a,pad_size_b),(pad_size_a,pad_size_b)),'constant')
        else:
            start = int((H*scale - H) * 0.5)
            out_image = resize_image[start:(start+H),start:(start+W),:]
            out_landmark2d = resize_landmark2d - np.array([start,start])
            out_landmark3d = resize_landmark3d - np.array([start,start])
            out_seg = resize_seg[start:(start+H),start:(start+W),:]
            out_glassframe = resize_glassframe[start:(start+H),start:(start+W)]
        return out_image.astype(np.float32), out_landmark2d.astype(np.float32), out_landmark3d.astype(np.float32), out_seg.astype(np.float32), out_glassframe.astype(np.float32)

    @staticmethod
    def random_color_shift(image):
        # random color shift
        # image should be in [0,255]
        random_color = 0.8 + 0.4 * np.random.sample(3)
        random_color = np.clip(random_color, 0, 255.0/image.max(axis=(0,1)))
        image = (image * random_color).astype(np.float32)
        return image

    @staticmethod
    def parse(tfrecord, options):
        features = tf.parse_single_example(
            tfrecord,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                '2Dlandmark': tf.FixedLenFeature([68*2], tf.float32),
                '3Dlandmark': tf.FixedLenFeature([86*2], tf.float32),
                'segmentation': tf.FixedLenFeature([300*300], tf.int64),
            })

        data_size = 300 # input resolution
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.cast(tf.reshape(image, [data_size,data_size,3]), tf.float32)
        landmark2d = features['2Dlandmark']
        landmark2d = tf.reshape(landmark2d, [68,2])
        landmark3d = features['3Dlandmark']
        landmark3d = tf.reshape(landmark3d, [86,2])
        segmentation = features['segmentation']
        segmentation = tf.reshape(segmentation, [data_size,data_size])

        # create one-hot segmentation labels
        segmentation = DataSet.create_one_hot_seg_label(segmentation)

        # add glassframe segmentation
        glass_seg_image = segmentation[:,:,3]
        [glassframe] = tf.py_func(detect_glassframe, [image/255.0,glass_seg_image],[tf.float32])
        if options['aug']:
            image, landmark2d, landmark3d, segmentation, glassframe = \
                    tf.py_func(DataSet.augment, [image, landmark2d, landmark3d, segmentation, glassframe, options['color'], options['scale'], options['flip']],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
            image = tf.reshape(image, [300,300,3])
            glassframe = tf.reshape(glassframe, [300,300])
            landmark2d = tf.reshape(landmark2d, [68,2])
            landmark3d = tf.reshape(landmark3d, [86,2])
            segmentation = tf.reshape(segmentation, [300,300,19])

        return image, landmark2d, landmark3d, segmentation, glassframe

    @staticmethod
    def load(data_dir, batch_size, options):
        """
        Load (image, landmark2d, landmark3d, segmentation).

        :param:
            data_dir: string, tfrecord directory
            batch_size: int, batch size
            options: {'aug': bool,'color':bool,'scale':bool,'flip':bool}, to indicate augmentation options
        """
        filenames = sorted(tf.gfile.Glob(os.path.join(data_dir, '*.tfrecords')) + \
                tf.gfile.Glob(os.path.join(data_dir,'*','*.tfrecords')))
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if 'shard' in options:
            dataset = dataset.shard(options['shard'], options['work_index'])
        dataset = dataset.shuffle(buffer_size=len(filenames))
        dataset = dataset.flat_map(tf.data.TFRecordDataset)

        #batch process
        dataset = dataset.prefetch(buffer_size=batch_size*10)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=512)

        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            lambda value: DataSet.parse(value, options),batch_size=batch_size))
        iterator = dataset.make_one_shot_iterator()
        image, landmark2d, landmark3d, segmentation, glassframe = iterator.get_next()

        # reshape to get batch size
        screen_size = 300
        image = tf.reshape(image,[batch_size,screen_size,screen_size,3])
        landmark2d = tf.reshape(landmark2d,[batch_size,68,2])
        landmark3d = tf.reshape(landmark3d,[batch_size,86,2])
        segmentation = tf.reshape(segmentation,[batch_size,screen_size,screen_size,19])
        glassframe = tf.reshape(glassframe, [batch_size,screen_size,screen_size])

        return image, landmark2d, landmark3d, segmentation, glassframe


if __name__ == '__main__':

    ''' test training dataset '''
    #data_dir = '/home/haoxianzhang/only_pony_for_overfit_tfrecords'
    #batch_size = 10
    #options = {
    #        'aug': True,
    #        'color': True, # TODO: debug with color shift
    #        'scale': True,
    #        'flip': True
    #        }
    #image, landmark2d, landmark3d, segmentation, glassframe = DataSet.load(data_dir, batch_size, options)

    #with tf.Session() as sess:
    #    img, lmk2d, lmk3d, seg, glsfrm = sess.run([image, landmark2d, landmark3d, segmentation, glassframe])

    #    # save one sample for visualization
    #    img1 = img[0]
    #    lmk2d1 = lmk2d[0]
    #    lmk3d1 = lmk3d[0]
    #    seg1 = seg[0]
    #    glsfrm = glsfrm[0]
    #    seg_list = []
    #    for i in range(seg1.shape[-1]):
    #        seg = seg1[:,:,i] * i * 255.0 / 18.0
    #        seg_list.append(seg)
    #    seg_list = np.stack(seg_list, axis=2)
    #    seg1 = np.sum(seg_list, axis=2)

    #    cv2.imwrite('tmp_glassframe.png', glsfrm * 255)

    #    for x, y in lmk3d1:
    #        img1 = cv2.circle(img1,(x,y),1,(255,0,0),1)
    #    for x, y in lmk2d1:
    #        img1 = cv2.circle(img1,(x,y),1,(0,0,255),1)
    #    img1 = img1[:,:,::-1]
    #    cv2.imwrite('tmp.png', img1)
    #    cv2.imwrite('tmp_seg.png', seg1)


    ''' test synthetic dataset '''

    data_dir = '../tools/tmp'
    dataset = SynDataSet()
    res = dataset.load(data_dir,4,True)
    img = res[0]
    shp = res[1]
    exp = res[2]
    tex_contour = res[3]
    pose = res[-2]
    light = res[-1]
    with tf.Session() as sess:
        img_out = sess.run(img)
        print(img_out.shape, np.amax(img_out), np.amin(img_out))
        shp_out = sess.run(shp)
        print(shp_out.shape, np.amax(shp_out), np.amin(shp_out))
        exp_out = sess.run(exp)
        print(exp_out.shape, np.amax(exp_out), np.amin(exp_out))
        tex_contour_out = sess.run(tex_contour)
        print(tex_contour_out.shape, np.amax(tex_contour_out), np.amin(tex_contour_out))
        pose_out = sess.run(pose)
        print(pose_out.shape, np.amax(pose_out), np.amin(pose_out))
        light_out = sess.run(light)
        print(light_out.shape, np.amax(light_out), np.amin(light_out))



