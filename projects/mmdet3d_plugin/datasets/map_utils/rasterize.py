import cv2
import numpy as np

import torch

from shapely import affinity
from shapely.geometry import LineString, box

def get_distance_transform(masks, threshold=None):
    # masks: (3, 196, 200) np bool
    labels = (~masks).astype('uint8')
    distances = np.zeros(masks.shape, dtype=np.float32)
    for i, label in enumerate(labels):
        distances[i] = cv2.distanceTransform(label, cv2.DIST_L2, maskSize=5)
        # truncate to [0.0, 10.0] and invert values
        if threshold is not None:
            distances[i] = float(threshold) - distances[i]
            distances[i][distances[i] < 0.0] = 0.0
            distances[i] = distances[i] / threshold # normalize
        # cv2.normalize(distances[i], distances[i], 0, 1.0, cv2.NORM_MINMAX)
    return distances

def sample_pts_from_line(line, 
                         fixed_num=-1,
                         sample_dist=1,
                         normalize=False,
                         patch_size=None,
                         padding=False,
                         num_samples=250,):
    if fixed_num < 0:
        distances = np.arange(0, line.length+sample_dist, sample_dist) # line.length+sample_dist: includes endpoint
        sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    else:
        # fixed number of points, so distance is line.length / fixed_num
        distances = np.linspace(0, line.length, fixed_num)
        sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

    if normalize:
        sampled_points = sampled_points / np.array([patch_size[1], patch_size[0]])

    num_valid = len(sampled_points)

    if not padding or fixed_num > 0:
        # fixed num sample can return now!
        return sampled_points, num_valid

    # fixed distance sampling need padding!
    num_valid = len(sampled_points)

    if fixed_num < 0:
        if num_valid < num_samples:
            padding = np.zeros((num_samples - len(sampled_points), 2))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        else:
            sampled_points = sampled_points[:num_samples, :]
            num_valid = num_samples

        if normalize:
            sampled_points = sampled_points / np.array([patch_size[1], patch_size[0]])
            num_valid = len(sampled_points)

    return sampled_points, num_valid

def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2)) # (N, 2), cols, rows
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0)

    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    elif type == 'vertex':
        for coord in coords:
            cv2.circle(mask, coord, radius=0, color=idx, thickness=-1)
        # mask[ [coord[1] for coord in coords], [coord[0] for coord in coords] ] = 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(mask, [coords[i:]], False, color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class), thickness=thickness)
    return mask, idx


def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type='index', angle_class=36):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
            # new_line: vectors in BEV pixel coordinate
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString' or new_line.geom_type == 'GeometryCollection':
                for new_single_line in new_line.geoms:
                    map_mask, idx = mask_for_lines(new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask


def preprocess_map(gt_dict, # dict('gt_labels_3d'=torch.tensor, 'gt_bboxes_3d'=LiDARInstanceLines)
                   pc_range, # [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0] x y z
                   voxel_size, # [0.15, 0.15, 4] x y z
                   num_classes, # 3
                   sample_dist=1.5,
                   cell_size=8,
                   dt_threshold=None):
    gt_labels = gt_dict['gt_labels_3d']
    gt_instances = gt_dict['gt_bboxes_3d'].instance_list # torch.Tensor
    
    gt_vectors = []
    patch_size = [pc_range[4] - pc_range[1], pc_range[3] - pc_range[0]] # 60, 30
    canvas_size = [int(patch_size[0]/voxel_size[0]), int(patch_size[1]/voxel_size[1])] # 400, 200
    
    for gt_instance, gt_label in zip(gt_instances, gt_labels):
        pts, pts_num = sample_pts_from_line(gt_instance, patch_size=patch_size, sample_dist=sample_dist)
        gt_vectors.append({
            'pts': pts,
            'pts_num': pts_num,
            'type': int(gt_label)
        })
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in gt_vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(LineString(vector['pts'][:vector['pts_num']]))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    distance_masks = []
    vertex_masks = []
    for i in range(num_classes):
        distance_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, 1, 1)
        distance_masks.append(distance_mask)
        vertex_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, 1, 1, type='vertex')
        vertex_masks.append(vertex_mask)

    # canvas_size: tuple (int, int)
    # vertex_masks: 3, 200, 400
    vertex_masks = np.stack(vertex_masks)
    # vertex_masks = vertex_masks.max(0) # 200, 400
    C, H, W = vertex_masks.shape # 3, 200, 400
    Hc, Wc = int(H/cell_size), int(W/cell_size)
    vertex_masks = np.reshape(vertex_masks, [C, Hc, cell_size, Wc, cell_size]) # 3, Hc, 8, Wc, 8
    vertex_masks = np.transpose(vertex_masks, [0, 1, 3, 2, 4]) # 3, Hc, Wc, 8, 8
    vertex_masks = np.reshape(vertex_masks, [C, Hc, Wc, cell_size*cell_size]) # 3, Hc, Wc, 64
    vertex_masks = vertex_masks.transpose(0, 3, 1, 2) # 3, 64, Hc, Wc
    for c, vertex_mask in enumerate(vertex_masks): # for every class,
        vertex_sum = vertex_mask.sum(0) # number of vertex in each cell, [Hc, Wc]
        # find cell with more then one vertex
        rows, cols = np.where(vertex_sum > 1)
        # N == len(rows) == len(cols)
        if len(rows):
            multi_vertex = vertex_mask[:, [row for row in rows], [col for col in cols]].transpose(1, 0) # N, 64
            index, depth = np.where(multi_vertex > 0)
            nums_multi_vertex = np.histogram(index, bins=len(rows), range=(0, len(rows)))[0]
            select = np.random.randint(nums_multi_vertex)
            nums_cum = np.insert(np.cumsum(nums_multi_vertex[:-1]), 0, 0)
            select_cum = select + nums_cum
            remove_index = np.delete(index, select_cum)
            remove_depth = np.delete(depth, select_cum)
            multi_vertex[[i for i in remove_index], [d for d in remove_depth]] = 0
            vertex_masks[c, :, [row for row in rows], [col for col in cols]] = multi_vertex
    vertex_sum = vertex_masks.sum(1) # number of vertex in each cell, 3, Hc, Wc
    assert np.max(vertex_sum) <= 1, f"max(vertex_sum) expected less than 1, but got: {np.max(vertex_sum)}" # make sure one vertex per cell
    # # randomly select one vertex and remove all others
    # dust = np.zeros_like(vertex_sum[0], dtype='uint8') # Hc, Wc
    # dust[vertex_sum == 0] = 1
    # dust = np.expand_dims(dust, axis=1) # 3, 1, Hc, Wc
    # vertex_masks = np.concatenate((vertex_masks, dust), axis=1) # 3, 65, Hc, Wc
    # assert np.min(vertex_masks.sum(1)) == 1

    distance_masks = np.stack(distance_masks)

    distance_masks = distance_masks != 0
    
    outs = {
        'gt_vectors': gt_vectors,
        'distance_transform': get_distance_transform(distance_masks, dt_threshold),
        'vertex_mask': vertex_masks
    }

    return outs


def rasterize_map(vectors, patch_size, canvas_size, num_classes, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append((LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    masks = []
    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        masks.append(map_mask)

    return np.stack(masks), confidence_levels
