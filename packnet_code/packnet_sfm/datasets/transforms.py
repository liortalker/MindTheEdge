# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC), based on:
# https://github.com/TRI-ML/packnet-sfm - Toyota Research Institute

import cv2
from functools import partial

from packnet_code.packnet_sfm.datasets.augmentations import resize_image, resize_sample, resize_depth, \
    duplicate_sample, colorjitter_sample, to_tensor_sample, crop_sample, crop_sample_input, resize_depth_preserve

from packnet_code.packnet_sfm.utils.depth import augment_depth_values

from packnet_code.packnet_sfm.utils.misc import parse_crop_borders

########################################################################################################################

def train_transforms(sample, image_shape, jittering, crop_train_borders, lidar_scale, lidar_add, lidar_drop_rate):
    """
    Training data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    crop_train_borders : tuple (left, top, right, down)
        Border for cropping

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(crop_train_borders) > 0:
        borders = parse_crop_borders(crop_train_borders, sample['rgb'].size[::-1])
        sample = crop_sample(sample, borders)
    if len(image_shape) > 0:
        # sample = resize_sample(sample, sample['rgb'].size[::-1])
        sample = resize_sample(sample, image_shape)
    sample = duplicate_sample(sample)
    if len(jittering) > 0:
        sample = colorjitter_sample(sample, jittering)
    if len(lidar_scale) > 0 and len(lidar_add) > 0:
        # sample = augment_depth_values(sample, lidar_scale, lidar_add, lidar_drop_rate)
        sample['input_depth'] = augment_depth_values(sample['input_depth'], lidar_scale, lidar_add, lidar_drop_rate)
    sample = to_tensor_sample(sample)
    return sample


def validation_transforms(sample, image_shape, crop_eval_borders):
    """
    Validation data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    crop_eval_borders : tuple (left, top, right, down)
        Border for cropping

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(crop_eval_borders) > 0:
        borders = parse_crop_borders(crop_eval_borders, sample['rgb'].size[::-1])
        sample = crop_sample_input(sample, borders)

    # just make sure it is an exponent of 2**5
    val_image_shape = list(sample['rgb'].size)
    # val_image_shape = image_shape
    if not val_image_shape[0] % 32 == 0:
        val_image_shape[0] = val_image_shape[0] - val_image_shape[0] % 32
    if not val_image_shape[1] % 32 == 0:
        val_image_shape[1] = val_image_shape[1] - val_image_shape[1] % 32
    val_image_shape = tuple([val_image_shape[1], val_image_shape[0]])
    sample['rgb'] = resize_image(sample['rgb'], val_image_shape)
    if 'input_depth' in sample:
        sample['input_depth'] = resize_depth_preserve(sample['input_depth'], val_image_shape)
    if 'edge' in sample:
        sample['edge'] = cv2.resize(sample['edge'], (val_image_shape[1], val_image_shape[0]))
    for i in range(1, 6):
        edge_str = 'edge_' + str(i)
        if edge_str in sample:
            sample[edge_str] = cv2.resize(sample[edge_str],
                                          (int(val_image_shape[1] / 2 ** i), int(val_image_shape[0] / 2 ** i)))
    if 'rgb_edge' in sample:
        sample['rgb_edge'] = cv2.resize(sample['rgb_edge'], (val_image_shape[1], val_image_shape[0]))

    sample = to_tensor_sample(sample)
    return sample



def test_transforms(sample, image_shape, crop_eval_borders):
    """
    Test data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(crop_eval_borders) > 0:
        borders = parse_crop_borders(crop_eval_borders, sample['rgb'].size[::-1])
        sample = crop_sample_input(sample, borders)
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)
        if 'input_depth' in sample:
            sample['input_depth'] = resize_depth(sample['input_depth'], image_shape)
    sample = to_tensor_sample(sample)
    return sample

def get_transforms(mode, image_shape, jittering, crop_train_borders,
                   crop_eval_borders, lidar_scale, lidar_add, lidar_drop_rate, **kwargs):
    """
    Get data augmentation transformations for each split

    Parameters
    ----------
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the data augmentation transformations
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    crop_train_borders : tuple (left, top, right, down)
        Border for cropping
    crop_eval_borders : tuple (left, top, right, down)
        Border for cropping

    Returns
    -------
        XXX_transform: Partial function
            Data augmentation transformation for that mode
    """
    if mode == 'train':
        return partial(train_transforms,
                       image_shape=image_shape,
                       jittering=jittering,
                       crop_train_borders=crop_train_borders,
                       lidar_scale=lidar_scale,
                       lidar_add=lidar_add,
                       lidar_drop_rate=lidar_drop_rate)
    elif mode == 'validation':
        return partial(validation_transforms,
                       crop_eval_borders=crop_eval_borders,
                       image_shape=image_shape)
    elif mode == 'test':
        return partial(test_transforms,
                       crop_eval_borders=crop_eval_borders,
                       image_shape=image_shape)
    else:
        raise ValueError('Unknown mode {}'.format(mode))

########################################################################################################################

