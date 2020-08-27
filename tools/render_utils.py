"""
A utility class to project 3D mesh into camera view, and render 3D mesh into 2D images.

We provide interface for two different camera models, i.e., ortho and perspective model.

We provide interface for two different lighting models, i.e., spherical harmonics model and phong model.

"""
import tensorflow as tf
import numpy as np
import cv2
import sys
from tools.const import OrthoCam
from tools.rasterize_triangles import rasterize_clip_space


class Shader(object):
    def __init__(self):
        pass

    @staticmethod
    def _harmonics(ver_norm):
        """ compute the spherical harmonics function for 3D vertices.
        :param:
            ver_norm: [batch, N, 3], vertex normal

        :return:
            H: [batch, 9], 2-order harmonic basis
        """
        x, y, z = tf.split(ver_norm, 3, -1)
        x2 = tf.square(x)
        y2 = tf.square(y)
        z2 = tf.square(z)
        xy = x * y
        yz = y * z
        xz = x * z
        PI = np.pi

        l0 = np.sqrt(1.0 / (4 * PI)) * tf.ones_like(x)
        l1x = np.sqrt(3.0 / (4 * PI)) * x
        l1y = np.sqrt(3.0 / (4 * PI)) * y
        l1z = np.sqrt(3.0 / (4 * PI)) * z
        l2xy = np.sqrt(15.0 / (4 * PI)) * xy
        l2yz = np.sqrt(15.0 / (4 * PI)) * yz
        l2xz = np.sqrt(15.0 / (4 * PI)) * xz
        l2z2 = np.sqrt(5.0 / (16 * PI)) * (3 * z2 - 1)
        l2x2_y2 = np.sqrt(15.0 / (16 * PI)) * (x2 - y2)
        H = tf.concat(
            [l0, l1z, l1x, l1y, l2z2, l2xz, l2yz, l2x2_y2, l2xy],
            -1,
            name="hamonics_basis_order",
        )
        return H

    @staticmethod
    def sh_shader(normals, background_images, sh_coefficients, diffuse_colors):
        """
        render mesh into image space and return all intermediate results.
        :param:
            normals: [batch,H,W,3], vertex normals in image space
            alphas: [batch,H,W,1], alpha channels
            background_images: [batch,H,W,3], background images for rendering results
            sh_coefficient: [batch,27], 2-order SH coefficient
            diffuse_colors: [batch,H,W,3], vertex colors in image space

        sh_coefficient: [batch_size, 27] spherical harmonics coefficients.
        """

        batch_size, image_height, image_width = [s.value for s in normals.shape[:-1]]
        pixel_count = image_height * image_width

        init_para_illum = tf.constant([0.3,0.3] + [0] * 7, tf.float32, name="init_illum")
        init_para_illum = tf.reshape(
            init_para_illum, [1, 9], name="init_illum_reshape"
        )
        init_para_illum = tf.concat(
            [init_para_illum] * 3, axis=1, name="init_illum_concat"
        )
        sh_coefficients = sh_coefficients + init_para_illum  # batch x 27

        if batch_size is None:
            batch_size = diffuse_colors.get_shape().as_list()[0]
        if batch_size is None:
            batch_size = sh_coefficients.get_shape().as_list()[0]
        sh_kernels = tf.unstack(sh_coefficients, batch_size, axis=0)

        # use conv to replace multiply for speed-up
        harmonic_output = Shader._harmonics(normals)
        harmonic_output_list = tf.split(harmonic_output, batch_size, axis=0)
        results = []
        for ho, shk in zip(harmonic_output_list, sh_kernels):
            shk = tf.reshape(tf.transpose(tf.reshape(shk, [3, 9])), [1, 1, 9, 3])
            res = tf.nn.conv2d(ho, shk, strides=[1, 1, 1, 1], padding="VALID")
            results.append(res)
        shading = tf.concat(results, axis=0)

        rgb_images = shading * diffuse_colors

        return rgb_images, shading


class Projector(object):
    def __init__(self):
        pass

    @staticmethod
    def _ortho_trans_clip(imageW):
        """
        transform into clip space according to orthogonal camera parameters.
        :param:
            imageW: float, image width
        :return:
            ortho_transforms: [batch, 4, 4], a orthogonal transform matrix into clip space
        """
        f = float(1.0 / (imageW * 0.5))
        ortho_transforms = tf.constant(
            [
                f,
                0.0,
                0.0,
                0.0,
                0.0,
                f,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0 / 10000,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=tf.float32,
            name="ortho_trans",
        )
        ortho_transforms = tf.reshape(
            ortho_transforms, [1, 4, 4], name="ortho_trans_reshape"
        )
        return ortho_transforms

    @staticmethod
    def _rgbd_camera_clip(fx, fy, cx, cy):
        fx_div_cx = tf.reshape(fx / cx, [1, 1])
        fy_div_cy = tf.reshape(fy / cy, [1, 1])
        ones = tf.ones(shape=[1, 1], dtype=tf.float32)
        zeros = tf.zeros(shape=[1, 1], dtype=tf.float32)
        M = tf.concat(
            [
                fx_div_cx,
                zeros,
                zeros,
                zeros,
                zeros,
                fy_div_cy,
                zeros,
                zeros,
                zeros,
                zeros,
                -ones,
                -0.005 * ones,
                zeros,
                zeros,
                -ones,
                zeros,
            ],
            axis=1,
        )
        M = tf.reshape(M, [1, 4, 4])
        return M


    @staticmethod
    def _ortho_trans_image(ver_xyzw, imageW, imageH):
        """
        Transform clip-space vertices into image space using orthogonal transform
        :param:
            ver_xyzw: [batch, N, 4], clip-space vertices in (x,y,z,w) form
            imageW: float, image width
            imageH: float, image height

        :return:
            trans_xyz: [batch, N, 3], transformed (x,y,z)
        """
        x, y, z, w = tf.split(ver_xyzw, 4, axis=-1)
        x = (x + 1.0) * imageW / 2.0
        y = (y + 1.0) * imageH / 2.0
        ver = tf.concat([x, imageH - y, z], axis=-1, name="ver_concat")
        return ver


    @staticmethod
    def calc_trans_matrix(para_pose, screen_size):
        """
        Calculate transformation matrix based on pose parameters and camera option.

        :param:
            para_pose: [batch, 6], pose paramters

        :return:
            T: [batch, 4, 4], transformation matrix
        """
        pitch, yaw, roll, tx, ty, sth = tf.split(
            para_pose, 6, axis=1, name="pose_split"
        )
        cos_x = tf.cos(pitch, name="cos_pitch")
        sin_x = tf.sin(pitch, name="sin_pitch")
        cos_y = tf.cos(yaw, name="cos_yaw")
        sin_y = tf.sin(yaw, name="sin_yaw")
        cos_z = tf.cos(roll, name="cos_roll")
        sin_z = tf.sin(roll, name="sin_roll")
        zeros = tf.zeros_like(sin_z, name="zeros")
        ones = tf.ones_like(sin_z, name="ones")

        # compute rotation matrices
        rx = tf.concat(
            [ones, zeros, zeros, zeros, cos_x, sin_x, zeros, -1 * sin_x, cos_x],
            axis=1,
            name="rx",
        )
        rx = tf.reshape(rx, [-1, 3, 3], name="rx_reshape")
        ry = tf.concat(
            [cos_y, zeros, -1 * sin_y, zeros, ones, zeros, sin_y, zeros, cos_y],
            axis=1,
            name="ry",
        )
        ry = tf.reshape(ry, [-1, 3, 3], name="ry_reshape")
        rz = tf.concat(
            [cos_z, sin_z, zeros, -1 * sin_z, cos_z, zeros, zeros, zeros, ones],
            axis=1,
            name="rz",
        )
        rz = tf.reshape(rz, [-1, 3, 3], name="rz_reshape")

        scale_ratio = 1.5 # constant value to make face to be large in the image
        R = tf.matmul(tf.matmul(rx, ry), rz, name="R")
        R = R * tf.expand_dims(tf.exp(sth), 1) * screen_size * scale_ratio / 300
        translation = tf.expand_dims(
            tf.concat([tx, ty, ones * float(OrthoCam["tz_init"]), ones], axis=1), 2, name="translation"
        )
        w = tf.expand_dims(tf.concat([zeros, zeros, zeros], axis=1), 1, name="w")
        T = tf.concat([R, w], axis=1, name="Tmat")
        T = tf.concat([T, translation], axis=2, name="Tmat2")
        return T


    @staticmethod
    def project(ver_xyz, M, ver_tri, tri, imageW, imageH, cam_config, var_scope_name):
        """
        transform local space vertices to world space and project to image space.

        :param:
            ver_xyz: [batch, N, 3], local-space vertices
            M: [batch, 4, 4], transformation matrix from local to world
            tri: [M, 3], triangle definition
            imageW: float, image width
            imageH: float, image height
            cam_config: dict, camera parameters
            var_scope_name: variable scope for shared variables with render process

        :return:
            image_xyz: [batch, N, 3], image-space vertices
        """
        with tf.variable_scope(var_scope_name, reuse=tf.AUTO_REUSE):
            Mclip = Projector._ortho_trans_clip(imageW)

            ver_w = tf.ones_like(ver_xyz[:, :, 0:1], name="ver_w")
            ver_xyzw = tf.concat([ver_xyz, ver_w], axis=2, name="ver_xyzw")

            Mclip = tf.concat(
                [Mclip] * ver_w.get_shape().as_list()[0], axis=0, name="Mclip"
            )

            cam_xyzw = tf.matmul(ver_xyzw, M, transpose_b=True, name="cam_xyzw")
            clip_xyzw = tf.matmul(cam_xyzw, Mclip, transpose_b=True, name="clip_xyzw")

            cam_xyz = cam_xyzw[:, :, :3]
            ver_normal, ver_contour_mask, tri_normal = Projector.get_ver_norm(cam_xyz, ver_tri, tri, var_scope_name)

        image_xyz = Projector._ortho_trans_image(clip_xyzw, imageW, imageH)

        return image_xyz, ver_normal, ver_contour_mask, tri_normal


    @staticmethod
    def get_ver_norm(ver_xyz, ver_tri, tri, scope_name):
        """
        Compute vertex normals and vertex contour mask(for 2d lmk loss).

        :param:
            ver_xyz: [batch, N, 3], vertex geometry
            tri: [M, 3], mesh triangles definition

        :return:
            ver_normals: [batch, N, 3], vertex normals
            ver_contour_mask: [batch, N, 1], vertex contour mask, indicating whether the vertex is on the contour
        """
        batch_size, n_ver, n_channels = ver_xyz.get_shape().as_list()
        n_ver_tri_pair = ver_tri.get_shape().as_list()[0]

        n_tri = tri.get_shape().as_list()[0]
        v1_idx, v2_idx, v3_idx = tf.unstack(tri, 3, axis=-1)
        v1 = tf.gather(ver_xyz, v1_idx, axis=1, name="v1_tri")
        v2 = tf.gather(ver_xyz, v2_idx, axis=1, name="v2_tri")
        v3 = tf.gather(ver_xyz, v3_idx, axis=1, name="v3_tri")

        EPS = 1e-8
        tri_normals = tf.cross(v2 - v1, v3 - v1)
        tri_visible = tf.cast(tf.greater(tri_normals[:, :, 2:], float(EPS)), tf.float32)

        to_ver_ids, from_tri_ids = tf.split(ver_tri, 2, axis=1)
        tmp_ver_tri_normals = tf.gather(tri_normals, tf.squeeze(from_tri_ids), axis=1)
        tmp_ver_tri_visible = tf.gather(tri_visible, tf.squeeze(from_tri_ids), axis=1)
        tmp_ver_tri_counts = tf.ones([batch_size, n_ver_tri_pair, 1], tf.float32)

        batch_indices = tf.reshape(
                        tf.tile(tf.expand_dims(tf.range(batch_size),axis=1),[1,n_ver_tri_pair]),
                        [batch_size, n_ver_tri_pair, 1], name='batch_indices')
        to_ver_ids = tf.tile(tf.expand_dims(to_ver_ids, axis=0), [batch_size,1,1])
        batch_to_ver_ids = tf.concat([batch_indices, to_ver_ids], axis=2)
        ver_normals = tf.scatter_nd(
                batch_to_ver_ids, tmp_ver_tri_normals, shape=[batch_size, n_ver, 3])
        ver_normals = tf.nn.l2_normalize(ver_normals, dim=2)

        ver_visible = tf.scatter_nd(
                batch_to_ver_ids, tmp_ver_tri_visible, shape=[batch_size, n_ver, 1])

        cond1 = tf.less(ver_visible, float(1.0))
        cond2 = tf.greater(ver_visible, float(0.0))
        ver_contour_mask = tf.cast(
            tf.logical_and(cond1, cond2),
            tf.float32,
            name="ver_votes_final",
        )
        return ver_normals, ver_contour_mask, tri_normals





    @staticmethod
    def get_ver_norm_bk(ver_xyz, ver_tri, tri, scope_name):
        """
        Compute vertex normals and vertex contour mask(for 2d lmk loss).

        :param:
            ver_xyz: [batch, N, 3], vertex geometry
            tri: [M, 3], mesh triangles definition

        :return:
            ver_normals: [batch, N, 3], vertex normals
            ver_contour_mask: [batch, N, 1], vertex contour mask, indicating whether the vertex is on the contour
        """
        batch_size, n_ver, n_channels = ver_xyz.get_shape().as_list()
        n_ver_tri_pair = ver_tri.get_shape().as_list()[0]

        n_tri = tri.get_shape().as_list()[0]
        v1_idx, v2_idx, v3_idx = tf.unstack(tri, 3, axis=-1)
        v1 = tf.gather(ver_xyz, v1_idx, axis=1, name="v1_tri")
        v2 = tf.gather(ver_xyz, v2_idx, axis=1, name="v2_tri")
        v3 = tf.gather(ver_xyz, v3_idx, axis=1, name="v3_tri")

        EPS = 1e-8
        tri_normals = tf.cross(v2 - v1, v3 - v1)
        tri_visible = tf.cast(tf.greater(tri_normals[:, :, 2:], float(EPS)), tf.float32)

        with tf.variable_scope(scope_name, tf.AUTO_REUSE):

            var_sum_of_tri_normals = tf.get_variable(
                    'ver_sum_of_tri_normals',
                    [batch_size, n_ver, 3],
                    tf.float32,
                    tf.zeros_initializer(),
                    trainable=False
                    )

            var_sum_of_tri_visible = tf.get_variable(
                    'ver_sum_of_tri_visible',
                    [batch_size, n_ver, 1],
                    tf.float32,
                    tf.zeros_initializer(),
                    trainable=False
                    )

            var_sum_of_tri_counts = tf.get_variable(
                    'ver_sum_of_counts',
                    [batch_size, n_ver, 1],
                    tf.float32,
                    tf.zeros_initializer(),
                    trainable=False
                    )

        init_sum_of_tri_normals = tf.zeros_like(var_sum_of_tri_normals)
        init_sum_of_tri_visible = tf.zeros_like(var_sum_of_tri_visible)
        init_sum_of_tri_counts = tf.zeros_like(var_sum_of_tri_counts)
        assign_op = tf.group([
            tf.assign(var_sum_of_tri_normals, init_sum_of_tri_normals),
            tf.assign(var_sum_of_tri_visible, init_sum_of_tri_visible),
            tf.assign(var_sum_of_tri_counts, init_sum_of_tri_counts)
            ], name='assign_op')

        with tf.control_dependencies([assign_op]):
            to_ver_ids, from_tri_ids = tf.split(ver_tri, 2, axis=1)
            tmp_ver_tri_normals = tf.gather(tri_normals, tf.squeeze(from_tri_ids), axis=1)
            tmp_ver_tri_visible = tf.gather(tri_visible, tf.squeeze(from_tri_ids), axis=1)
            tmp_ver_tri_counts = tf.ones([batch_size, n_ver_tri_pair, 1], tf.float32)

            print(tmp_ver_tri_normals.get_shape().as_list())

            batch_indices = tf.reshape(
                            tf.tile(tf.expand_dims(tf.range(batch_size),axis=1),[1,n_ver_tri_pair]),
                            [batch_size, n_ver_tri_pair, 1], name='batch_indices')
            to_ver_ids = tf.tile(tf.expand_dims(to_ver_ids, axis=0), [batch_size,1,1])
            batch_to_ver_ids = tf.concat([batch_indices, to_ver_ids], axis=2)
            var_sum_of_tri_normals = tf.scatter_nd_add(var_sum_of_tri_normals,
                    batch_to_ver_ids, tmp_ver_tri_normals)
            var_sum_of_tri_visible = tf.scatter_nd_add(var_sum_of_tri_visible,
                    batch_to_ver_ids, tmp_ver_tri_visible)
            var_sum_of_tri_counts = tf.scatter_nd_add(var_sum_of_tri_counts,
                    batch_to_ver_ids, tmp_ver_tri_counts)
            ver_normals = tf.div(var_sum_of_tri_normals, var_sum_of_tri_counts + EPS)
            ver_normals = tf.nn.l2_normalize(ver_normals)

            ver_visible = tf.div(var_sum_of_tri_visible, var_sum_of_tri_counts + EPS)

            cond1 = tf.less(ver_visible, float(1.0))
            cond2 = tf.greater(ver_visible, float(0.0))
            ver_contour_mask = tf.cast(
                tf.logical_and(cond1, cond2),
                tf.float32,
                name="ver_votes_final",
            )
            return ver_normals, ver_contour_mask



    @staticmethod
    def get_ver_norm_bk(ver_xyz, tri):
        """
        Compute vertex normals and vertex contour mask(for 2d lmk loss).

        :param:
            ver_xyz: [batch, N, 3], vertex geometry
            tri: [M, 3], mesh triangles definition

        :return:
            ver_normals: [batch, N, 3], vertex normals
            ver_contour_mask: [batch, N, 1], vertex contour mask, indicating whether the vertex is on the contour
        """
        n_tri = tri.get_shape().as_list()[0]
        v1_idx, v2_idx, v3_idx = tf.unstack(tri, 3, axis=-1)
        v1 = tf.gather(ver_xyz, v1_idx, axis=1, name="v1_tri")
        v2 = tf.gather(ver_xyz, v2_idx, axis=1, name="v2_tri")
        v3 = tf.gather(ver_xyz, v3_idx, axis=1, name="v3_tri")

        EPS = 1e-8
        tri_normals = tf.cross(v2 - v1, v3 - v1)
        #tri_normals = tf.div(
        #    tri_normals,
        #    (tf.norm(tri_normals, axis=-1, keep_dims=True) + EPS),
        #    name="norm_tri",
        #)
        #tri_normals = tf.nn.l2_normalize(tri_normals, dim=-1)
        tmp = tf.tile(
            tf.expand_dims(tri_normals, 2),
            [1, 1, 3, 1],
            name="tri_normals_tile"
        ) # per vertex attribute
        # per_vertex: information for each vertex in triangle
        tri_normals_per_vertex = tf.reshape(tmp, [-1, 3], name="tri_normals_reshape")
        tri_visible_per_vertex = tf.cast(tf.greater(tri_normals_per_vertex[:, 2:], float(EPS)), tf.float32)
        tri_one_per_vertex = tf.ones_like(tri_visible_per_vertex, name="tri_cnts")

        B = v1.get_shape().as_list()[0]  # batch size
        batch_indices = tf.reshape(
            tf.tile(tf.expand_dims(tf.range(B), axis=1), [1, n_tri * 3]),
            [-1],
            name="batch_indices",
        )
        tri_inds = tf.stack(
            [batch_indices, tf.concat([tf.reshape(tri, [n_tri * 3])] * B, axis=0)],
            axis=1,
            name="tri_inds",
        )

        ver_shape = ver_xyz.get_shape().as_list()

        ver_normals = tf.get_variable(
            shape=ver_shape,
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            name="ver_norm",
            trainable=False,
        )

        # refresh normal per iteration
        init_normals = tf.zeros(shape=ver_shape, dtype=tf.float32, name="init_normals")
        assign_op = tf.assign(ver_normals, init_normals, name="ver_normal_assign")
        with tf.control_dependencies([assign_op]):
            ver_normals = tf.scatter_nd_add(
                ver_normals, tri_inds, tri_normals_per_vertex, name="ver_normal_scatter"
            )
            #ver_normals = ver_normals / (
            #    tf.norm(ver_normals, axis=2, keep_dims=True, name="ver_normal_norm")
            #    + EPS
            #)
            ver_normals = tf.nn.l2_normalize(ver_normals, dim=-1)

        ver_visible = tf.get_variable(
            shape=ver_shape[:-1] + [1],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            name="ver_vote",
            trainable=False,
        )
        ver_tri_cnt = tf.get_variable(
            shape=ver_shape[:-1] + [1],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            name="ver_cnt",
            trainable=False,
        )
        init_values = tf.zeros(
            shape=ver_shape[:-1] + [1], dtype=tf.float32, name="init_votes"
        )
        # find the visible boundary
        assign_op2 = tf.assign(ver_visible, init_values, name="ver_votes_assign")
        assign_op3 = tf.assign(ver_tri_cnt, init_values, name="ver_cnts_assign")
        with tf.control_dependencies([assign_op2, assign_op3]):
            ver_visible = tf.scatter_nd_add(
                ver_visible, tri_inds, tri_visible_per_vertex, name="ver_vote_scatter"
            )
            ver_tri_cnt = tf.scatter_nd_add(
                ver_tri_cnt, tri_inds, tri_one_per_vertex, name="ver_cnt_scatter"
            )
            ver_visible_ratio = ver_visible / (ver_tri_cnt + EPS)

            cond1 = tf.less(ver_visible_ratio, float(1.0))
            cond2 = tf.greater(ver_visible_ratio, float(0.0))
            ver_contour_mask = tf.cast(
                tf.logical_and(cond1, cond2),
                tf.float32,
                name="ver_votes_final",
            )

        return ver_normals, ver_contour_mask

    @staticmethod
    def sh_render(
        ver_xyz,
        ver_rgb,
        ver_attrs,
        ver_tri,
        tri,
        M,
        para_light,
        background,
        cam_config,
        var_scope_name,
    ):
        """
        render the local-space vertices into image-space according to camera model.

        :param:
            ver_xyz: [batch, N, 3], local-space verices in (x,y,z) form
            ver_rgb: [batch, N, 3], vertex color (r,g,b)
            ver_attrs: [batch, N, ?], vertex attributes, e.g., vertex normals
            tri: [M, 3], mesh triangle definition
            M: [batch, 4, 4], transformation to world space
            para_light: [batch, 27], spherical harmonics parameters
            background: [batch, 300, 300, 3], background images
            cam_config: dict, camera parameters
            var_scope_name: variable scope name to share variables with projection

        :return:
            rgb_image: [batch, H, W, 3], rendered image
            norm_image: [batch, H, W, 3], vertex normal rasterized
            diffuse_image: [batch, H, W, 3], vertex color rasterized
            shading_image: [batch, H, W, 3], shading image
            attrs_image: [batch, H, W, ?], attributes rasterized
        """
        _, imageH, imageW, _ = background.get_shape().as_list()
        with tf.variable_scope(var_scope_name, reuse=tf.AUTO_REUSE):
            Mclip = Projector._ortho_trans_clip(imageW)

            ver_w = tf.ones_like(ver_xyz[:, :, 0:1], name="ver_w")
            ver_xyzw = tf.concat([ver_xyz, ver_w], axis=2)

            Mclip = tf.concat([Mclip] * ver_w.get_shape().as_list()[0], axis=0)

            cam_xyzw = tf.matmul(ver_xyzw, M, transpose_b=True)
            clip_xyzw = tf.matmul(cam_xyzw, Mclip, transpose_b=True)

            cam_xyz = cam_xyzw[:, :, :3]
            clip_xyz = clip_xyzw[:,:,:3]
            ver_normal, ver_contour_mask, tri_normal = Projector.get_ver_norm(cam_xyz, ver_tri, tri, var_scope_name)

        aug_ver_attrs = tf.concat([ver_rgb, cam_xyz, ver_normal, ver_attrs], axis=2)
        attrs = rasterize_clip_space(
            clip_xyz, aug_ver_attrs, tri, imageW, imageH, -1.0
        )

        diffuse_image = attrs[:, :, :, :3]
        depth_image = attrs[:, :, :, 5:6]
        norm_image = attrs[:, :, :, 6:9]
        attrs_image = attrs[:, :, :, 9:]

        background = tf.reverse(background, axis=[1])
        rgb_image, shading_image = Shader.sh_shader(
            norm_image, background, para_light, diffuse_image
        )

        rgb_image = tf.clip_by_value(tf.reverse(rgb_image, axis=[1]), 0, 1)
        depth_image = tf.reverse(depth_image, axis=[1])
        norm_image = tf.reverse(norm_image, axis=[1])
        diffuse_image = tf.clip_by_value(tf.reverse(diffuse_image, axis=[1]), 0, 1)
        shading_image = tf.clip_by_value(tf.reverse(shading_image, axis=[1]), 0, 1)
        attrs_image = tf.clip_by_value(tf.reverse(attrs_image, axis=[1]), 0, 1)

        rgb_image = tf.reshape(rgb_image, [-1, imageH, imageW, 3])
        depth_image = tf.reshape(depth_image, [-1, imageH, imageW, 1])
        norm_image = tf.reshape(norm_image, [-1, imageH, imageW, 3])

        return rgb_image, depth_image, norm_image, attrs_image


