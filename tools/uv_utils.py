from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tools.rasterize_triangles import rasterize_clip_space


def uv_renderer(uv_coords,
                triangles,
                uv_width,
                uv_height,
                uv_vertex_attrs):
  if len(uv_coords.shape) != 3:
    raise ValueError('uv_coords must have shape [batch_size, vertex_count, 2].')
  batch_size = uv_vertex_attrs.shape[0].value
  if len(uv_vertex_attrs.shape) != 3:
    raise ValueError(
        'uv_vertex_colors must have shape [batch_size, vertex_count, 3].')

  # normalize uv_coords to be ranging within [-1, 1]
  u, v, z = tf.split(uv_coords, 3, axis=-1)
  u = u * 2 - 1
  v = v * 2 - 1
  w = tf.ones_like(u)
  normalized_uv_coords = tf.concat([v,u,z,w], axis=-1)

  identity_matrices = tf.expand_dims(tf.eye(4), 0)
  identity_matrices = tf.tile(identity_matrices, [batch_size, 1, 1])

  renders = rasterize_clip_space(
      normalized_uv_coords, uv_vertex_attrs, triangles,
      uv_width, uv_height, [-1] * uv_vertex_attrs.shape[2].value)

  return renders, alphas



def convert_uv_attrs_into_ver(
        uv_attrs,
        uv_coords,
        triangles,
        uv_width,
        uv_height):
  if len(uv_coords.shape) != 3:
    raise ValueError('uv_coords must have shape [batch_size, vertex_count, 2].')
  batch_size = uv_attrs.shape[0].value
  if len(uv_attrs.shape) != 4:
    raise ValueError(
        'uv_attrs must have shape [batch_size, uv_size, uv_size, 3].')

  # normalize uv_coords to be ranging within [-1, 1]
  U, V = tf.split(uv_coords, 2, axis=-1)
  U = tf.cast(U, tf.int32)
  V = tf.cast(V, tf.int32)

  n_channels = uv_attrs.get_shape().as_list()[-1]
  batch_size, n_ver, _ = uv_coords.get_shape().as_list()
  batch_indices = tf.tile(
          tf.reshape(tf.range(batch_size),[batch_size, 1, 1]),
          [1,n_ver,1])
  batch_proj_coords = tf.concat([batch_indices, V, U], axis=2)
  batch_proj_coords = tf.reshape(batch_proj_coords, [batch_size * n_ver, 3]) # [[batch, u, v]] * batch_size * n_ver

  ver_attrs = tf.reshape(tf.gather_nd(uv_attrs, batch_proj_coords), [batch_size, n_ver, n_channels])
  return ver_attrs


def convert_ver_attrs_into_uv(
        ver_attrs,
        uv_coords,
        triangles,
        uv_width,
        uv_height):
  if len(uv_coords.shape) != 3:
    raise ValueError('uv_coords must have shape [batch_size, vertex_count, 2].')
  batch_size = ver_attrs.shape[0].value
  if len(ver_attrs.shape) != 3:
    raise ValueError(
        'ver_colors must have shape [batch_size, vertex_count, 3].')

  # normalize uv_coords to be ranging within [-1, 1]
  u, v, z = tf.split(uv_coords, 3, axis=-1)
  u = u * 2 - 1
  v = v * 2 - 1
  w = tf.ones_like(u)
  normalized_uv_coords = tf.concat([v,u,z,w], axis=-1)

  n_channels = ver_attrs.get_shape().as_list()[-1]

  renders = rasterize_clip_space(
      normalized_uv_coords, ver_attrs, triangles, uv_width,
      uv_height, [-1] * ver_attrs.shape[2].value)
  renders = tf.reshape(renders, [batch_size, uv_height, uv_width, n_channels])

  renders = tf.where(tf.is_nan(renders),
          tf.zeros_like(renders),
          renders)

  return renders


class TopoUV2Ver(object):

    def __init__(self, uv_size, uv_face_mask, mode='sparse'):
        self.uv_size = uv_size
        self.mode = mode
        self.cut_off = 0.5
        self.uv_face_mask = uv_face_mask
        map_uv_to_ver_id, ver_id_to_uv = self._get_uv_to_ver_id(uv_size)
        triangles = self._get_triangles(map_uv_to_ver_id)
        self.ver_neighbors = self._get_vertex_neighbors(triangles)
        self.ver_tri = self._get_vertex_tri_neighbors(triangles)
        self.ver_uv = ver_id_to_uv
        self.triangles = triangles
        self.tri_tri = self._get_tri_tri_neighbors(triangles)

    def _get_uv_to_ver_id(self, uv_size):

        map_uv_to_ver_id = {}
        ver_id_to_uv = []
        ver_id = 0

        if self.mode == 'sparse':
            for i_h in range(0, uv_size, 2):
                if i_h % 4 == 0:
                    start_w = 0
                else:
                    start_w = 1

                for i_w in range(start_w, uv_size, 2):
                    ver_uv = (i_h, i_w)
                    if self.uv_face_mask[i_h, i_w] < self.cut_off:
                        continue
                    map_uv_to_ver_id[ver_uv] = ver_id
                    ver_id_to_uv.append([i_h, i_w])
                    ver_id += 1
        else:
            for i_h in range(0, uv_size):
                for i_w in range(0, uv_size):
                    if self.uv_face_mask[i_h, i_w] < self.cut_off:
                        continue
                    ver_uv = (i_h, i_w)
                    map_uv_to_ver_id[ver_uv] = ver_id
                    ver_id_to_uv.append([i_h, i_w])
                    ver_id += 1

        ver_id_to_uv = np.array(ver_id_to_uv, np.float32)
        return map_uv_to_ver_id, ver_id_to_uv

    def _get_triangles(self, map_uv_to_ver_id):
        triangles = []

        if self.mode == 'sparse':

            for i_h in range(0, self.uv_size, 2):
                if i_h % 4 == 0:
                    start_w = 0
                else:
                    start_w = 1

                for i_w in range(start_w, self.uv_size, 2):
                    if i_h < 0 or i_h+2 >= self.uv_size or i_w < 0 or i_w + 2 >= self.uv_size:
                        continue

                    if self.uv_face_mask[i_h, i_w] < self.cut_off or self.uv_face_mask[i_h, i_w+2] < self.cut_off or self.uv_face_mask[i_h+2,i_w+1] < self.cut_off:
                        continue
                    v1_uv = (i_h, i_w)
                    v2_uv = (i_h, i_w+2)
                    v3_uv = (i_h+2, i_w+1)

                    v1 = map_uv_to_ver_id[v1_uv]
                    v2 = map_uv_to_ver_id[v2_uv]
                    v3 = map_uv_to_ver_id[v3_uv]

                    tri = [v1, v3, v2]
                    triangles.append(tri)

                for i_w in range(start_w, self.uv_size, 2):
                    if i_h < 0 or i_h+2 >= self.uv_size or i_w-1 < 0 or i_w + 1 >= self.uv_size:
                        continue

                    if self.uv_face_mask[i_h, i_w] < self.cut_off or self.uv_face_mask[i_h+2, i_w+1] < self.cut_off or self.uv_face_mask[i_h+2,i_w-1] < self.cut_off:
                        continue
                    v1_uv = (i_h, i_w)
                    v2_uv = (i_h+2,i_w+1)
                    v3_uv = (i_h+2,i_w-1)

                    v1 = map_uv_to_ver_id[v1_uv]
                    v2 = map_uv_to_ver_id[v2_uv]
                    v3 = map_uv_to_ver_id[v3_uv]

                    tri = [v1, v3, v2]
                    triangles.append(tri)
        else:

            for i_h in range(self.uv_size - 1):

                for i_w in range(self.uv_size - 1):
                    # (i,j), (i,j+1), (i+1,j+1)
                    # (i,j), (i+1,j+1), (i+1,j)

                    if self.uv_face_mask[i_h, i_w] < self.cut_off \
                            or self.uv_face_mask[i_h,i_w+1] < self.cut_off \
                            or self.uv_face_mask[i_h+1,i_w] < self.cut_off \
                            or self.uv_face_mask[i_h+1,i_w+1] < self.cut_off:
                        continue

                    v1_uv = (i_h, i_w)
                    v2_uv = (i_h, i_w+1)
                    v3_uv = (i_h+1, i_w+1)
                    v4_uv = (i_h+1, i_w)

                    v1 = map_uv_to_ver_id[v1_uv]
                    v2 = map_uv_to_ver_id[v2_uv]
                    v3 = map_uv_to_ver_id[v3_uv]
                    v4 = map_uv_to_ver_id[v4_uv]

                    tri = [v1, v3, v2]
                    triangles.append([v1, v3, v2])
                    triangles.append([v1, v4, v3])

        triangles = np.array(triangles, np.int32)

        return triangles

    def _get_vertex_neighbors(self, triangles):
        vertex_neighbor_set = set()
        for tri in triangles:
            v1 = tri[0]
            v2 = tri[1]
            v3 = tri[2]
            vertex_neighbor_set.add((v1,v2))
            vertex_neighbor_set.add((v2,v1))
            vertex_neighbor_set.add((v1,v3))
            vertex_neighbor_set.add((v3,v1))
            vertex_neighbor_set.add((v2,v3))
            vertex_neighbor_set.add((v3,v2))
        vertex_neighbors = np.array(list(vertex_neighbor_set), np.int32)
        return vertex_neighbors

    def _get_tri_tri_neighbors(self, triangles):

        # get the neighboring relationship of triangles
        edge_to_triangles = {}
        for idx, tri in enumerate(triangles):
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
        return tri_pairs



    def _get_vertex_tri_neighbors(self, triangles):
        vertex_tri_set = set()
        for i, tri in enumerate(triangles):
            v1 = tri[0]
            v2 = tri[1]
            v3 = tri[2]
            vertex_tri_set.add((v1,i))
            vertex_tri_set.add((v2,i))
            vertex_tri_set.add((v1,i))
            vertex_tri_set.add((v3,i))
            vertex_tri_set.add((v2,i))
            vertex_tri_set.add((v3,i))
        vertex_tri_set = np.array(list(vertex_tri_set), np.int32)
        return vertex_tri_set


def remesh_uv_to_ver(uv_attrs, ver_uv):
    # UV to ver
    # ver_uv: tf tensor
    batch_size, uv_size, _, n_channels = uv_attrs.get_shape().as_list()
    ver_uv = tf.tile(
            tf.expand_dims(ver_uv, 0),
            [batch_size, 1, 1])
    _, n_ver, _ = ver_uv.get_shape().as_list()
    batch_indices = tf.reshape(
                        tf.tile(tf.expand_dims(tf.range(batch_size),axis=1),[1,n_ver]),
                        [batch_size, n_ver, 1], name='batch_indices')
    ver_uv = tf.cast(ver_uv, tf.int32)
    batch_indices = tf.reshape(
            tf.concat([batch_indices, ver_uv], axis=2),
            [batch_size * n_ver, 3])

    ver_attrs = tf.reshape(
            tf.gather_nd(uv_attrs, batch_indices),
            [batch_size, n_ver, n_channels])

    ver_attrs = tf.where(tf.is_nan(ver_attrs),
            tf.zeros_like(ver_attrs),
            ver_attrs)
    return ver_attrs


def unwrap_screen_into_uv(images, screen_coords, tri, ver_uv_index, uv_size):

    # prepare UV maps
    imageH, imageW = images.get_shape().as_list()[1:3]
    n_channels = images.get_shape().as_list()[-1]
    batch_size, n_ver, _ = screen_coords.get_shape().as_list()
    batch_indices = tf.tile(
            tf.reshape(tf.range(batch_size),[batch_size, 1, 1]),
            [1,n_ver,1])
    proj_x, proj_y = tf.split(screen_coords, 2, axis=2)
    proj_x = tf.clip_by_value(proj_x, 0, imageW-1)
    proj_y = tf.clip_by_value(proj_y, 0, imageH-1)
    proj_x = tf.cast(proj_x, tf.int32)
    proj_y = tf.cast(proj_y, tf.int32)
    batch_screen_coords = tf.concat([batch_indices, proj_y, proj_x], axis=2)
    batch_screen_coords = tf.reshape(batch_screen_coords, [batch_size * n_ver, 3])

    ver_colors = tf.reshape(
            tf.gather_nd(images, batch_screen_coords),
            [batch_size, n_ver, n_channels]
            )

    uv_colors = convert_ver_attrs_into_uv(ver_colors, ver_uv_index, tri, uv_size, uv_size)
    return uv_colors



