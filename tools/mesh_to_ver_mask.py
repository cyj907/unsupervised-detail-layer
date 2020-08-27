# mesh to vertex mask
from ply import read_ply
import numpy as np

wo_eyebrow_path = '../resources/mesh_wo_eyebrow.ply'
wo_nose_path = '../resources/mesh_wo_nose.ply'


def from_tri_to_vertices(tri):
    v1, v2, v3 = np.split(tri, 3, axis=-1)
    vertices = np.squeeze(np.concatenate([v1, v2, v3], axis=0))
    vertices = vertices.tolist()
    vertices = set(vertices)
    return vertices

def from_vertices_to_mask(vertices, max_vertex_id):

    ver_mask = np.zeros([max_ver_id])
    for v in vertices:
        ver_mask[v] = 1
    ver_mask = np.array(ver_mask)
    return ver_mask

wo_eyebrow_data = read_ply(wo_eyebrow_path)
wo_eyebrow_vertices = from_tri_to_vertices(np.array(wo_eyebrow_data['mesh']))
wo_nose_data = read_ply(wo_nose_path)
wo_nose_vertices = from_tri_to_vertices(np.array(wo_nose_data['mesh']))

max_ver_id = np.array(wo_eyebrow_data['points']).shape[0]
wo_eyebrow_mask = from_vertices_to_mask(wo_eyebrow_vertices, max_ver_id)
wo_nose_mask = from_vertices_to_mask(wo_nose_vertices, max_ver_id)

np.save('../resources/wo_eyebrow_mask.npy', wo_eyebrow_mask)
np.save('../resources/wo_nose_mask.npy', wo_nose_mask)

