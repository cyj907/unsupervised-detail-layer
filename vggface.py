""" a tensorflow version of VGG face model.
This snippet is modified from https://github.com/ZZUTK/Tensorflow-VGG-face.git
The pretrained weights are also downloaded from the same repo
"""
import tensorflow as tf
import numpy as np
import scipy.io
import tensorflow.contrib.layers as Layers

class VGGFace(object):

    def __init__(self, param_path, trainable=True):
        self.data = scipy.io.loadmat(param_path)
        self.trainable = trainable

    def encoder(self, input_maps, reuse=False):
        with tf.variable_scope('VGGFace') as scope:
            if reuse:
                scope.reuse_variables()
            # read meta info
            meta = self.data['meta']
            classes = meta['classes']
            class_names = classes[0][0]['description'][0][0]
            normalization = meta['normalization']
            average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
            image_size = np.squeeze(normalization[0][0]['imageSize'][0][0])
            input_maps = tf.image.resize_images(input_maps, (image_size[0], image_size[1]))

            # read layer info
            layers = self.data['layers']
            current = input_maps
            network = {}
            network['inputs'] = input_maps
            input_maps = input_maps - average_image
            for layer in layers[0]:
                name = layer[0]['name'][0][0]
                layer_type = layer[0]['type'][0][0]
                if layer_type == 'conv':
                    if name[:2] == 'fc':
                        padding = 'VALID'
                    else:
                        padding = 'SAME'
                    stride = layer[0]['stride'][0][0]
                    kernel, bias = layer[0]['weights'][0][0]
                    bias = np.squeeze(bias).reshape(-1)
                    kh, kw, kin, kout = kernel.shape

                    kernel_var = tf.get_variable(name=name+'_weight', dtype=tf.float32,initializer=tf.constant(kernel), trainable=self.trainable)
                    bias_var = tf.get_variable(name=name+'_bias',dtype=tf.float32,initializer=tf.constant(bias), trainable=self.trainable)

                    if name[:2] != 'fc':
                        conv = tf.nn.conv2d(current, kernel_var,
                                            strides=(1, stride[0], stride[0], 1), padding=padding)
                        current = tf.nn.bias_add(conv, bias_var)
                    else:
                        _, cur_h, cur_w, cur_c = current.get_shape().as_list()
                        flatten = tf.reshape(current, [-1, cur_h * cur_w * cur_c], name='flatten')
                        kernel_var = tf.reshape(kernel_var, [kh * kw * kin, kout], name='kernel_flat')
                        fc = tf.matmul(flatten, kernel_var)
                        current = fc + bias_var
                        current = tf.reshape(current, [-1,1,1,kout])

                elif layer_type == 'relu':
                    current = tf.nn.relu(current)
                elif layer_type == 'pool':
                    stride = layer[0]['stride'][0][0]
                    pool = layer[0]['pool'][0][0]
                    current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                             strides=(1, stride[0], stride[0], 1), padding='SAME')
                elif layer_type == 'softmax':
                    current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))

                network[name] = current
                print(name)

            return network, average_image, class_names

if __name__ == '__main__':
    vggface = VGGFace('../resources/vgg-face.mat')
    image_batch = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vggface.encoder(image_batch)
