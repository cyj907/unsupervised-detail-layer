import tensorflow as tf
import numpy as np
from vggface import VGGFace
import tools.basis_utils as basis_op
import tools.uv_utils as uv_op
from tools.render_utils import Projector
from tools.const import OrthoCam

MAX = 1e5

class CoarseModel(object):

    def __init__(self, vggpath='', basis3dmm=None, trainable=True):
        self.basis3dmm = basis3dmm
        self.model = VGGFace(vggpath, trainable)

    def encoder3DMM(self, imgs, reuse=False):

        with tf.variable_scope('CoarseModel', reuse=tf.AUTO_REUSE):
            layers, _, _ = self.model.encoder(imgs, reuse)

            z = tf.reshape(layers['fc7'], [-1, 4096])

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:

                para_shape = tf.layers.dense(z,
                    self.basis3dmm['bases_shape'].shape[0],
                    use_bias=False,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                    name='para_shape')

                para_exp = tf.layers.dense(z,
                    self.basis3dmm['bases_exp'].shape[0],
                    use_bias=False,
                    kernel_initializer=tf.zeros_initializer(),
                    name='para_exp')

                para_tex = tf.layers.dense(z,
                    self.basis3dmm['bases_tex'].shape[0],
                    use_bias=False,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                    name='para_tex')

                para_pose  = tf.layers.dense(z,
                    6,
                    use_bias=False,
                    kernel_initializer=tf.zeros_initializer(),
                    name='para_pose')

                para_illum = tf.layers.dense(z,
                    27,
                    use_bias=False,
                    kernel_initializer=tf.zeros_initializer(),
                    name='para_illum')

            return para_shape, para_exp, para_tex, para_pose, para_illum


class FineModel(object):

    def __init__(self):
        pass

    def generator(self, input_uv, coarse_uv, visible_mask):
        """ auto-encoder for uv maps.
        :param:
            :input_uv: unwrap input uv maps.
            :render_coarse_uv: unwrap render coarse uv maps
            :mask_uv: indicating the visible region in uv masks
        :return:
            :disp_z: displacement in z direction.
        """

        #gray_input_uv = tf.reduce_mean(input_uv, axis=-1, keepdims=True)
        #gray_coarse_uv = tf.reduce_mean(coarse_uv, axis=-1, keepdims=True)
        inputs = (input_uv - coarse_uv)

        #inputs = gray_input_uv - gray_coarse_uv

        # 0818 version (add sharpen filter)
        #kernel = tf.constant([-1,-1,-1,-1,8,-1,-1,-1,-1], tf.float32)
        #kernel = tf.reshape(kernel, [3,3,1,1])

        #################################

        with tf.variable_scope('FineModel') as scope:
            # TODO: add high-pass filter

            conv0 = tf.layers.conv2d(inputs, 16, [3,3], 1, padding='SAME') # 512 -> 512
            conv0_act = tf.nn.relu(conv0)

            #conv1 = tf.layers.conv2d(conv0_act, 32, [7,7], 2, padding='SAME') # 512 -> 256
            conv1 = tf.layers.conv2d(conv0_act, 32, [3,3], 2, padding='SAME') # 512 -> 256
            conv1_act = tf.nn.relu(conv1)

            #conv2 = tf.layers.conv2d(conv1_act, 64, [5,5], 2, padding='SAME') # 256 -> 128
            conv2 = tf.layers.conv2d(conv1_act, 64, [3,3], 2, padding='SAME') # 256 -> 128
            conv2_act = tf.nn.relu(conv2)

            #conv3 = tf.layers.conv2d(conv2_act, 128, [5,5], 2, padding='SAME') # 128 -> 64
            conv3 = tf.layers.conv2d(conv2_act, 128, [3,3], 2, padding='SAME') # 128 -> 64
            conv3_act = tf.nn.relu(conv3)

            #conv4 = tf.layers.conv2d(conv3_act, 256, [5,5], 2, padding='SAME') # 64 -> 32
            conv4 = tf.layers.conv2d(conv3_act, 256, [3,3], 2, padding='SAME') # 64 -> 32
            conv4_act = tf.nn.relu(conv4)

            #conv5 = tf.layers.conv2d(conv4_act, 512, [5,5], 2, padding='SAME') # 32 -> 16
            conv5 = tf.layers.conv2d(conv4_act, 512, [3,3], 2, padding='SAME') # 32 -> 16
            conv5_act = tf.nn.relu(conv5)

            conv6 = tf.layers.conv2d(conv5_act, 512, [3,3], 2, padding='SAME') # 16 -> 8
            conv6_act = tf.nn.relu(conv6)

            conv7 = tf.layers.conv2d(conv6_act, 512, [3,3], 2, padding='SAME') # 8 -> 4
            conv7_act = tf.nn.relu(conv7)

            conv8 = tf.layers.conv2d(conv7_act, 512, [3,3], 2, padding='SAME') # 4 -> 2
            conv8_act = tf.nn.relu(conv8)

            conv9 = tf.layers.conv2d(conv8_act, 512, [3,3], 2, padding='SAME') # 2 -> 1
            conv9_act = tf.nn.relu(conv9)

            deconv9 = tf.layers.conv2d_transpose(conv9_act, 512, [3,3], 2, padding='SAME') # 4 -> 8
            deconv9 = tf.nn.relu(deconv9) + conv8_act

            deconv8 = tf.layers.conv2d_transpose(deconv9, 512, [3,3], 2, padding='SAME') # 4 -> 8
            deconv8 = tf.nn.relu(deconv8) + conv7_act

            deconv7 = tf.layers.conv2d_transpose(deconv8, 512, [3,3], 2, padding='SAME') # 4 -> 8
            deconv7 = tf.nn.relu(deconv7) + conv6_act

            deconv6 = tf.layers.conv2d_transpose(deconv7, 512, [3,3], 2, padding='SAME') # 4 -> 8
            deconv6 = tf.nn.relu(deconv6) + conv5_act

            #deconv5 = tf.layers.conv2d_transpose(deconv6, 256, [5,5], 2, padding='SAME') # 16 -> 32
            deconv5 = tf.layers.conv2d_transpose(deconv6, 256, [3,3], 2, padding='SAME') # 16 -> 32
            deconv5 = tf.nn.relu(deconv5) + conv4_act

            #deconv4 = tf.layers.conv2d_transpose(deconv5, 128, [5,5], 2, padding='SAME') # 32 -> 64
            deconv4 = tf.layers.conv2d_transpose(deconv5, 128, [3,3], 2, padding='SAME') # 32 -> 64
            deconv4 = tf.nn.relu(deconv4) + conv3_act

            #deconv3 = tf.layers.conv2d_transpose(deconv4, 64, [5,5], 2, padding='SAME') # 64 -> 128
            deconv3 = tf.layers.conv2d_transpose(deconv4, 64, [3,3], 2, padding='SAME') # 64 -> 128
            deconv3 = tf.nn.relu(deconv3) + conv2_act

            #deconv2 = tf.layers.conv2d_transpose(deconv3, 32, [5,5], 2, padding='SAME') # 128 -> 256
            deconv2 = tf.layers.conv2d_transpose(deconv3, 32, [3,3], 2, padding='SAME') # 128 -> 256
            deconv2 = tf.nn.relu(deconv2) + conv1_act

            #deconv1 = tf.layers.conv2d_transpose(deconv2, 16, [7,7], 2, padding='SAME') # 256 -> 512
            deconv1 = tf.layers.conv2d_transpose(deconv2, 16, [3,3], 2, padding='SAME') # 256 -> 512
            deconv1 = tf.nn.relu(deconv1) + conv0_act

            deconv0 = tf.layers.conv2d_transpose(deconv1, 1, [3,3], 1, padding='SAME') # 256 -> 512
            depth_disp_uv = tf.nn.tanh(deconv0)
            #depth_disp_uv = deconv0

        return depth_disp_uv


def build_model(images, vggpath='', basis3dmm=None, trainable=True, is_fine_model=False):

    # build coarse model
    images224 = tf.image.resize_images(images, (224, 224))
    coarse_model = CoarseModel(vggpath, basis3dmm, trainable)
    para_shape, para_exp, para_tex, para_pose, para_illum = coarse_model.encoder3DMM(images224)

    # get vertices (geometry and texture)
    ver_xyz = basis_op.get_geometry(basis3dmm, para_shape, para_exp)
    ver_rgb = basis_op.get_texture(basis3dmm, para_tex)
    coarse_tri = tf.constant(basis3dmm['tri'])
    coarse_ver_tri = tf.constant(basis3dmm['ver_tri'], tf.int32)

    if is_fine_model:
        screen_size = 300
        images = tf.image.resize_images(images, (screen_size, screen_size))
    else:
        screen_size = 300

    # build transform matrix from pose para
    batch_size, imageW, imageH = images.get_shape().as_list()[0:3]
    trans = Projector.calc_trans_matrix(para_pose, screen_size)
    proj_xyz, ver_norm, ver_mask_contour, tri_norm = Projector.project(
            ver_xyz, trans, coarse_ver_tri, coarse_tri, imageW, imageH, OrthoCam, "projector"
            )

    # render images
    ver_mask_face = tf.expand_dims(
            tf.concat([basis3dmm['mask_face'].astype(np.float32)] * batch_size, axis=0, name='face_mask'),
            -1)
    ver_mask_wo_eye = tf.expand_dims(
            tf.concat([basis3dmm['mask_wo_eye'].astype(np.float32)] * batch_size, axis=0, name='wo_eye_mask'),
            -1)
    ver_mask_wo_nose = tf.expand_dims(
            tf.concat([basis3dmm['mask_wo_nose'].astype(np.float32)] * batch_size, axis=0, name='wo_nose_mask'),
            -1)
    ver_mask_wo_eyebrow = tf.expand_dims(
            tf.concat([basis3dmm['mask_wo_eyebrow'].astype(np.float32)] * batch_size, axis=0, name='wo_eyebrow_mask'),
            -1)
    ver_mask_attrs = tf.concat([ver_mask_face, ver_mask_wo_eye, ver_mask_wo_nose, ver_mask_wo_eyebrow], -1)

    render_rgb, render_depth, render_normal, render_mask = Projector.sh_render(
            ver_xyz,
            ver_rgb / 255.0,
            ver_mask_attrs,
            coarse_ver_tri,
            coarse_tri,
            trans,
            para_illum,
            images / 255.0,
            OrthoCam,
            "projector"
            )
    tf.summary.image('render_normal', render_normal, max_outputs=1)

    render_rgb = render_rgb[:,:,:,:3] * 255.0  # we don't use the last alpha channel

    render_mask_face = render_mask[:,:,:,0:1]
    render_mask_wo_eye = render_mask[:,:,:,1:2]
    render_mask_wo_nose = render_mask[:,:,:,2:3]
    render_mask_wo_eyebrow = render_mask[:,:,:,3:4]
    render_mask_photo = render_mask_face * render_mask_wo_eye
    render_rgb = render_rgb * render_mask_face + images * (1 - render_mask_face)

    # get_landmark
    landmark_ind = np.squeeze(basis3dmm['keypoints'].astype(np.int32))
    pred_landmarks = tf.gather(proj_xyz[:,:,:2], landmark_ind, axis=1)

    coarse_results = {}
    coarse_results['para'] = {
            'shape': para_shape,
            'exp': para_exp,
            'tex': para_tex,
            'pose': para_pose,
            'illum': para_illum
            }
    coarse_results['ver'] = {
            'proj_xy': proj_xyz[:,:,:2],
            'xyz': ver_xyz,
            'rgb': ver_rgb,
            'normal': ver_norm,
            'face_mask': ver_mask_face,
            'wo_eye_mask': ver_mask_wo_eye,
            'wo_eyebrow_mask': ver_mask_wo_eyebrow,
            'wo_nose_mask': ver_mask_wo_nose,
            'contour_mask': ver_mask_contour,
            'landmark': pred_landmarks
            }
    coarse_results['screen'] = {
            'rgb': render_rgb,
            'depth': render_depth,
            'face_mask': render_mask_face,
            'wo_eye_mask': render_mask_wo_eye,
            'wo_eyebrow_mask': render_mask_wo_eyebrow,
            'wo_nose_mask': render_mask_wo_nose,
            'photo_mask': render_mask_photo
            }

    ret_results = {}
    ret_results['coarse'] = coarse_results

    if is_fine_model:

        # TODO: debug masks
        ver_mask_visible = tf.cast(tf.greater(ver_norm[:,:,2:3], 0.0), tf.float32) * ver_mask_face

        ver_mask_disp = ver_mask_wo_eye * ver_mask_wo_nose * ver_mask_wo_eyebrow

        # prepare uv indices for unwrapping
        ver_uv_index = basis_op.add_z_to_UV(basis3dmm)
        ver_uv_index = tf.tile(ver_uv_index[np.newaxis,...], [batch_size,1,1])

        uv_size = 512

        proj_coords = proj_xyz[:,:,:2]
        uv_inputs = uv_op.unwrap_screen_into_uv(images, proj_coords, coarse_tri, ver_uv_index, uv_size)
        uv_normal = uv_op.unwrap_screen_into_uv(render_normal, proj_coords, coarse_tri, ver_uv_index, uv_size)
        uv_rgb = uv_op.unwrap_screen_into_uv(render_rgb, proj_coords, coarse_tri, ver_uv_index, uv_size)

        uv_xyz = uv_op.convert_ver_attrs_into_uv(ver_xyz, ver_uv_index, coarse_tri, uv_size, uv_size)
        uv_rgb_tmp = uv_op.convert_ver_attrs_into_uv(ver_rgb, ver_uv_index, coarse_tri, uv_size, uv_size)
        uv_mask_face = uv_op.convert_ver_attrs_into_uv(ver_mask_face, ver_uv_index, coarse_tri, uv_size, uv_size)
        uv_mask_disp = uv_op.convert_ver_attrs_into_uv(ver_mask_disp, ver_uv_index, coarse_tri, uv_size, uv_size)
        uv_mask_visible = uv_op.convert_ver_attrs_into_uv(ver_mask_visible, ver_uv_index, coarse_tri, uv_size, uv_size)

        # debug visible mask
        tf.summary.image('uv_mask_visible', tf.cast(uv_mask_visible * 255, tf.uint8), max_outputs=1)
        tf.summary.image('render_depth', render_depth, max_outputs=1)

        uv_rgb = tf.clip_by_value(uv_rgb, 0, 255)
        uv_rgb_tmp = tf.clip_by_value(uv_rgb_tmp, 0, 255)
        uv_inputs = tf.clip_by_value(uv_inputs, 0, 255)
        uv_mask_face = tf.clip_by_value(uv_mask_face, 0, 1)
        uv_mask_disp = tf.clip_by_value(uv_mask_disp, 0, 1)
        uv_mask_visible = tf.clip_by_value(uv_mask_visible, 0, 1)

        # build fine model
        fine_model = FineModel()
        uv_disp_depth = fine_model.generator((uv_inputs - 127.5) / 127.5, (uv_rgb - 127.5) / 127.5, uv_mask_visible)
        fine_topo = uv_op.TopoUV2Ver(uv_size, basis3dmm['uv_face_mask'], 'dense')
        new_ver_uv = tf.constant(fine_topo.ver_uv)
        fine_tri = tf.constant(fine_topo.triangles)
        fine_tri_tri = tf.constant(fine_topo.tri_tri, tf.int32)
        fine_ver_tri = tf.constant(fine_topo.ver_tri, tf.int32)
        tf.summary.image('uv_disp_depth', uv_disp_depth, max_outputs=1)

        ver_rgb_new = uv_op.remesh_uv_to_ver(uv_rgb_tmp, new_ver_uv)
        ver_xyz_new = uv_op.remesh_uv_to_ver(uv_xyz, new_ver_uv)
        ver_disp_new = uv_op.remesh_uv_to_ver(uv_disp_depth, new_ver_uv)
        ver_mask_face_new = uv_op.remesh_uv_to_ver(uv_mask_face, new_ver_uv)
        ver_norm_new = uv_op.remesh_uv_to_ver(uv_normal, new_ver_uv)
        ver_mask_disp_new = uv_op.remesh_uv_to_ver(uv_mask_disp, new_ver_uv)
        ver_mask_visible_new = uv_op.remesh_uv_to_ver(uv_mask_visible, new_ver_uv)

        ver_x, ver_y, ver_z = tf.split(ver_xyz_new, 3, axis=-1)
        ver_xyz_fine = tf.concat([ver_x, ver_y, ver_z + ver_disp_new * ver_mask_disp_new], axis=-1)
        _, ver_norm_fine, _, tri_norm_fine = Projector.project(
            ver_xyz_fine, trans, fine_ver_tri, fine_tri, imageW, imageH, OrthoCam, "projector_fine"
            )

        render_rgb_fine, render_depth_fine, render_normal_fine, render_mask_fine = Projector.sh_render(
            ver_xyz_fine,
            ver_rgb_new / 255.0,
            ver_mask_face_new,
            fine_ver_tri,
            fine_tri,
            trans,
            para_illum,
            images / 255.0,
            OrthoCam,
            "projector_fine"
            )
        render_rgb_fine = render_rgb_fine[:,:,:,:3] * 255.0  # we don't use the last alpha channel
        render_rgb_fine = render_rgb_fine * render_mask_face * render_mask_wo_eye + images * (1 - render_mask_wo_eye * render_mask_face)
        tf.summary.image('render_depth_fine', render_depth_fine, max_outputs=1)
        tf.summary.image('render_normal_fine', render_normal_fine, max_outputs=1)

        fine_results = {}

        fine_results['ver'] = {}
        fine_results['ver']['xyz'] = ver_xyz_fine
        fine_results['ver']['adj'] = tf.constant(fine_topo.ver_neighbors)
        fine_results['ver']['rgb'] = ver_rgb_new
        fine_results['ver']['normal'] = ver_norm_fine
        fine_results['ver']['coarse_normal'] = ver_norm_new
        fine_results['ver']['disp_depth'] = ver_disp_new
        fine_results['ver']['mask_face'] = ver_mask_face_new
        fine_results['ver']['mask_disp'] = ver_mask_disp_new # displacement region

        fine_results['tri'] = {}
        fine_results['tri']['index'] = fine_tri
        fine_results['tri']['normal'] = tri_norm_fine
        fine_results['tri']['adj'] = fine_tri_tri

        fine_results['uv'] = {}
        fine_results['uv']['mask_face'] = uv_mask_face
        fine_results['uv']['mask_visible'] = uv_mask_visible
        fine_results['uv']['rgb'] = uv_rgb
        fine_results['uv']['input'] = uv_inputs
        fine_results['uv']['disp_depth'] = uv_disp_depth
        fine_results['uv']['xyz_new'] = uv_xyz
        uv_x, uv_y, uv_z = tf.split(uv_xyz, 3, axis=-1)
        uv_z = uv_z + uv_disp_depth
        uv_xyz_fine = tf.concat([uv_x, uv_y, uv_z], axis=-1)
        fine_results['uv']['xyz'] = uv_xyz_fine
        tf.summary.image('uv_xyz', uv_xyz_fine, max_outputs=1)

        fine_results['screen'] = {}
        fine_results['screen']['rgb'] = render_rgb_fine
        fine_results['screen']['inputs'] = images
        fine_results['screen']['depth'] = render_depth_fine
        fine_results['screen']['mask_face'] = render_mask_fine

        ret_results['fine'] = fine_results

    return ret_results

