# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC), based on:
# https://github.com/TRI-ML/packnet-sfm - Toyota Research Institute

import argparse
import numpy as np
import os
import torch
import cv2

from glob import glob
from cv2 import imwrite
import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# font = {'family': 'normal',
#         'weight': 'bold',
#         'size': 18}
# matplotlib.rc('font', **font)

from packnet_code.packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_code.packnet_sfm.datasets.augmentations import resize_image, resize_depth, resize_depth_preserve, to_tensor
from packnet_code.packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_code.packnet_sfm.utils.image import load_image
from packnet_code.packnet_sfm.utils.config import parse_test_file
from packnet_code.packnet_sfm.utils.load import set_debug

from packnet_code.packnet_sfm.datasets.kitti_dataset import read_png_depth
from packnet_code.packnet_sfm.utils.tools import non_max_suppression, hysteresis
from packnet_code.packnet_sfm.datasets.gta_dataset import read_lidar, process_lidar

sigmoid = torch.nn.Sigmoid()


def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM inference script')
    parser.add_argument('--config', type=str, help='Input file (.yaml)')
    args = parser.parse_args()
    assert args.config.endswith(('.yaml')), \
        'You need to provide a .yaml file'
    return args

def main(args):
    args = parse_args()

    # Parse arguments
    config, state_dict = parse_test_file(args.config)

    image_shape = config.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)
    if not os.path.exists(config.save.folder):
        os.makedirs(config.save.folder)

    if config.datasets.test.normals:
        if not os.path.exists(config.save.folder + '/normals'):
            os.makedirs(config.save.folder + '/normals')
    dtype = None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()

    split_file_path = config.datasets.test.split[0]
    split_file = open(split_file_path, 'r')
    split_file_line = split_file.readlines()

    files = [x.split('\n')[0].split(' ')[0] for x in split_file_line]

    lidar_files = [x.split('\n')[0].split(' ')[3] for x in split_file_line]

    if config.model.depth_net.input_channels == 4:
        rgb_edge_files = [x.split('\n')[0].split(' ')[5] for x in split_file_line]
    else:
        rgb_edge_files = [None for _ in lidar_files]  # torch.ones_like(lidar_files) * None

    K = np.array([960, 0, 960, 0, 960, 540, 0, 0, 1]).reshape([3, 3])
    counter = 0

    for fn, lidar_fn, rgb_edge_fn in zip(files, lidar_files, rgb_edge_files):

        infer_and_save_depth(
            fn, lidar_fn, rgb_edge_fn, K, config.save.folder, model_wrapper, config, image_shape, None, False, '', counter)
        counter += 1

        print('Processed image ' + str(counter))

    save_split_list(files, lidar_files, config.save.folder, config.save.folder + '/normals')

    print('-> Done!')


def save_split_list(rgb_files, lidar_files, save_folder_edges, save_folder_normals):

    filenames = [str(a).zfill(8) + '_lidar_000.png' for a in list(range(0,len(rgb_files)))]

    total_split_list =\
        [rgb_file + ' ' + lidar_file + ' ' + save_folder_edges + '/' + filename + ' ' + lidar_file + ' None None None ' + save_folder_normals + '/' + filename + '\n' for rgb_file, lidar_file, filename in zip(rgb_files, lidar_files, filenames)]

    f = open(save_folder_edges + '/rgb_lidar_edges_split.txt', 'w')
    f.writelines(total_split_list)
    f.close()

@torch.no_grad()
def infer_and_save_depth(input_file, lidar_input_file, rgb_edge_input_file, K, output_file, model_wrapper, config, image_shape, image_crop, half, save, counter):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : str
        Image file
    output_file : str
        Output file, or folder where the output will be saved
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        output_dir = output_file
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, str.zfill(str(counter), 8) + '.png')
        output_normals_file = output_dir + '/normals/' + str.zfill(str(counter), 8) + '.png'

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file)
    im_cols, im_rows = image.size

    if not image_shape is None:
        if im_cols != image_shape[0] or im_rows != image_shape[1]:
            image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)

    if rgb_edge_input_file is not None:
        if rgb_edge_input_file.split('.')[-1] == 'npy':
            rgb_edge_image = np.load(rgb_edge_input_file)
        elif rgb_edge_input_file.split('.')[-1] == 'png':
            rgb_edge_image = cv2.imread(rgb_edge_input_file)[:,:,0]
            rgb_edge_image = rgb_edge_image/255
        rgb_edge_image = to_tensor(rgb_edge_image).unsqueeze(0)

        if torch.cuda.is_available():
            rgb_edge_image = rgb_edge_image.to('cuda:{}'.format(rank()), dtype=dtype)
    else:
        rgb_edge_image = None

    # check if to infer multiscale
    if config.save.depth.multiscale:
        scales = 4
    else:
        scales = 1
        
    # Depth inference (returns predicted inverse depth)
    if config.datasets.test.is_infer_rgb:
        pred_inv_depth = model_wrapper.depth(image, rgb_edge = rgb_edge_image)['inv_depths'][0]

        for scale_idx in range(0,scales):
            if scales == 1:
                end_str = '_regular'
            else:
                end_str = '_regular_' + str.zfill(str(scale_idx),3)
            pred_depth_sigmoid = pred_inv_depth[scale_idx] / 2
            pred_numpy = pred_depth_sigmoid[0, 0, :, :].detach().cpu().numpy()
            if config.datasets.test.normals:
                sobelx = cv2.Sobel(pred_numpy, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(pred_numpy, cv2.CV_64F, 0, 1, ksize=5)
                # values [0,255], where the actual values are degrees of atan2 (linear transform from [-pi,pi])
                sobel_angle = np.arctan2(-sobely, sobelx)
                sobel_angle_255 = (((sobel_angle * (180 / np.pi) + 180) / 360) * 255).astype('uint8')
                imwrite(output_normals_file[:-4] + end_str + '.png', sobel_angle_255)
            if config.datasets.test.nms:
                pred_numpy = non_max_suppression(pred_numpy)
            if config.datasets.test.hysteresis:
                pred_numpy = hysteresis(pred_numpy)

            imwrite(output_file[:-4] + end_str + '.png', (pred_numpy) * 255)
            if config.save.depth.npz:
                np.save(output_file[:-4] + end_str + '.npy', pred_numpy)

    if not config.datasets.test.input_depth_type[0] == '' and config.datasets.test.is_infer_lidar:

        if lidar_input_file.split('.')[-1] == 'png':
            lidar_image = read_png_depth(lidar_input_file)
            lidar_image[lidar_image < 0.0] = 0.0

        elif lidar_input_file.split('.')[-1] == 'bin':
            raw_lidar_map = read_lidar(lidar_input_file)
            lidar_image = process_lidar(raw_lidar_map, K)

        elif lidar_input_file.split('.')[-1] == 'npy':
            lidar_image = np.load(lidar_input_file)

        # LT: why 200???
        lidar_image = lidar_image/200.0

        lidar_shape = lidar_image.shape
        if not image_shape is None:
            if lidar_shape[0] != image_shape[0] or lidar_shape[1] != image_shape[1]:
                lidar_image = resize_depth_preserve(lidar_image, image_shape)
        lidar_image = to_tensor(lidar_image).unsqueeze(0)

        if torch.cuda.is_available():
            lidar_image = lidar_image.to('cuda:{}'.format(rank()), dtype=dtype)

        pred_inv_depth_rgbd = model_wrapper.depth(image, lidar_image, rgb_edge=rgb_edge_image)['inv_depths'][0]

        for scale_idx in range(0,scales):
            if scales == 1:
                end_str = '_lidar'
            else:
                end_str = '_lidar_' + str.zfill(str(scale_idx), 3)

            pred_depth_rgbd_sigmoid = pred_inv_depth_rgbd[scale_idx] / 2
            pred_rgbd_numpy = pred_depth_rgbd_sigmoid[0, 0, :, :].detach().cpu().numpy()
            if config.datasets.test.normals:
                sobelx = cv2.Sobel(pred_rgbd_numpy, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(pred_rgbd_numpy, cv2.CV_64F, 0, 1, ksize=5)
                # values [0,255], where the actual values are degrees of atan2 (linear transform from [-pi,pi])
                sobel_angle = np.arctan2(-sobely, sobelx)
                sobel_angle_255 = (((sobel_angle * (180 / np.pi) + 180) / 360) * 255).astype('uint8')
                imwrite(output_normals_file[:-4] + end_str + '.png', sobel_angle_255)
            # non-max suppresion
            if config.datasets.test.nms:
                pred_rgbd_numpy = non_max_suppression(pred_rgbd_numpy)
            if config.datasets.test.hysteresis:
                pred_rgbd_numpy = hysteresis(pred_rgbd_numpy)

            imwrite(output_file[:-4] + end_str + '.png', (pred_rgbd_numpy) * 255)
            if config.save.depth.npz:
                np.save(output_file[:-4] + end_str + '.npy', pred_rgbd_numpy)
