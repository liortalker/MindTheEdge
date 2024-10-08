# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC). All rights reserved.

import cv2
import numpy as np
import PIL.Image as pil
from PIL import Image

def load_image(path):
    """
    Read an image using PIL

    Parameters
    ----------
    path : str
        Path to the image

    Returns
    -------d
    image : PIL.Image
        Loaded image
    """
    im = Image.open(path)
    if im.mode == 'RGBA':
        im = im.convert('RGB')

    return im

def chamfer_distance(im_pred, im_gt, mask=None, edge_to_edge_thresh=5):
    # a = np.array(([0, 1, 1, 1, 1],
    #               [0, 0, 1, 1, 1],
    #               [0, 1, 1, 1, 1],
    #               [0, 1, 1, 1, 0],
    #               [0, 1, 1, 0, 0]))

    if not mask is None:
        mask = np.repeat(np.expand_dims(mask.astype('float'),2), 3, axis=2)

    im_gt_norm = im_gt/255
    im_gt_norm[im_gt_norm > 0.5] = 1.0
    im_gt_norm[im_gt_norm <= 0.5] = 0.0
    if not mask is None:
        im_gt_norm = im_gt_norm * mask
    im_gt_uint = 1 - im_gt_norm.astype('uint8')
    gt_dist_im = ndimage.distance_transform_edt(im_gt_uint)

    im_pred_norm = im_pred/255
    im_pred_norm[im_pred_norm > 0.5] = 1.0
    im_pred_norm[im_pred_norm <= 0.5] = 0.0
    if not mask is None:
        im_pred_norm = im_pred_norm * mask

    c_dist = np.sum(gt_dist_im*im_pred_norm)/np.sum(im_pred_norm)

    if len(gt_dist_im) == 3:
        gt_dist_im_flatten = gt_dist_im[:,:,0].flatten()
        im_pred_flatten = im_pred_norm[:,:,0].flatten()
    else:
        gt_dist_im_flatten = gt_dist_im.flatten()
        im_pred_flatten = im_pred_norm.flatten()

    edges_cond = gt_dist_im_flatten[np.where(im_pred_flatten >= 0.5)[0]] < edge_to_edge_thresh
    percentage = np.sum(edges_cond)/np.sum(im_pred_flatten)

    edges_cond_reshaped = gt_dist_im_flatten.copy()
    edges_cond_reshaped[np.where(im_pred_flatten >= 0.5)[0]] = edges_cond
    edges_cond_reshaped[np.where(im_pred_flatten < 0.5)[0]] = -1

    edges_cond_reshaped = np.reshape(edges_cond_reshaped,gt_dist_im.shape)

    return c_dist, percentage, edges_cond_reshaped

def edge_from_depth(depth_gt, new_shape, name_edge_im, min_depth=0.0, max_depth=80.0, thresh_1=20, thresh_2=40, is_write_edge=True):
    depth_im = read_depth_file(depth_gt.split('\n')[0])

    if not new_shape is None:
        depth_resized = cv2.resize(depth_im, new_shape,
                                        interpolation=cv2.INTER_LINEAR)
    else:
        depth_resized = depth_im
    depth_mask = depth_resized < min_depth
    depth_resized[depth_mask] = min_depth
    depth_mask = depth_resized > max_depth
    depth_resized[depth_mask] = max_depth
    factor = 255.0/max_depth
    depth_resized_vis = depth_resized * factor
    depth_resized_vis = depth_resized_vis.astype(np.uint8)
    edge_im = cv2.Canny(depth_resized_vis, thresh_1, thresh_2)

    if is_write_edge:
        cv2.imwrite(name_edge_im, edge_im)

    return edge_im

def read_depth_file(file):

    if file.split('.')[-1] == 'png':
        return read_png_depth(file)
    elif file.split('.')[-1] == 'npy':
        return read_npy_depth(file)

def read_npy_depth(file):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)
    # return np.expand_dims(depth, axis=2)
    return depth

def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_image(file), dtype=int)
    # assert (np.max(depth_png) > 255), 'Wrong .png depth file'
    depth = depth_png.astype(np.float)
    # depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    # return np.expand_dims(depth, axis=2)
    return depth
