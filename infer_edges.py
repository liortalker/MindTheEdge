# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC), based on:
# https://github.com/TRI-ML/packnet-sfm - Toyota Research Institute

import argparse
import numpy as np
import os
import torch
import PIL.Image as pil
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.cm as cm

from glob import glob
from cv2 import imwrite
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
font = {'family': 'normal',
        'weight': 'bold',
        'size': 18}
matplotlib.rc('font', **font)
import sys

from packnet_code.packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_code.packnet_sfm.datasets.augmentations import resize_image, resize_depth_preserve, to_tensor
from packnet_code.packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_code.packnet_sfm.utils.image import load_image
from packnet_code.packnet_sfm.utils.config import parse_test_file
from packnet_code.packnet_sfm.utils.load import set_debug
from packnet_code.packnet_sfm.utils.depth import inv2depth, viz_inv_depth
from packnet_code.packnet_sfm.utils.logging import pcolor
from packnet_code.packnet_sfm.utils.save import save_paths_list
from packnet_code.packnet_sfm.datasets.kitti_dataset import read_png_depth, read_npz_depth
from packnet_code.packnet_sfm.datasets.gta_dataset import read_lidar, process_lidar

from packnet_code.packnet_sfm.utils.d3r import d3r

from eval_depth_edges import pr_evaluation, mean_recall_at_precision_range
from eval_depth import run_analysis, DensePredictionAnalyzer, DataLoader
import glob


def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM inference script')
    # parser.add_argument('file', type=str, help='Input file (.yaml)')
    # parser.add_argument('--checkpoint', type=str, help='Input file (.ckpt)')
    parser.add_argument('--config', type=str, help='Input file (.yaml)')
    args = parser.parse_args()
    # assert args.checkpoint.endswith(('.ckpt')), \
    #     'You need to provide a .ckpt file'
    assert args.config.endswith(('.yaml')), \
        'You need to provide a .yaml file'
    return args



def main(args):
    args = parse_args()
    # Initialize horovod
    # hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(args.config)

    image_shape = config.datasets.augmentation.image_shape

    crop_shape = config.datasets.augmentation.crop_eval_borders

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    # dtype = torch.float16 if args.half else None
    dtype = None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()

    split_file_path = config.datasets.test.split[0]
    split_file = open(split_file_path, 'r')
    split_file_line = split_file.readlines()

    # check if the test path is already the prefix of the test filenames. If so, do not concat
    files = [x.split('\n')[0].split(' ')[0] for x in split_file_line]

    lidar_files = [x.split('\n')[0].split(' ')[3] for x in split_file_line]

    if config.model.depth_net.input_channels == 4:
        rgb_edge_files = [x.split('\n')[0].split(' ')[5] for x in split_file_line]
    else:
        rgb_edge_files = [None]*len(lidar_files)


    # Process each file
    if not config.analysis.just_evaluate:
        counter = 0

        for fn, lidar_fn, rgb_edge_fn in zip(files, lidar_files, rgb_edge_files):

            infer_and_save_depth(
                fn, lidar_fn, rgb_edge_fn, config.save.folder, model_wrapper, config, image_shape, crop_shape, False, '', counter)

            counter += 1

        save_paths_list(files[rank()::world_size()], config.save.folder, "input_list.txt")
        pred_file_list = glob.glob(config.save.folder + '/*_regular.npy')
        save_paths_list(pred_file_list, config.save.folder, "pred_list.txt")
        pred_lidar_file_list = glob.glob(config.save.folder + '/*_lidar.npy')
        save_paths_list(pred_lidar_file_list, config.save.folder, "pred_lidar_list.txt")
        print('-> Done!')

    os.makedirs(config.save.folder + '/sfm_analysis', exist_ok=True)
    os.makedirs(config.save.folder + '/sfm_analysis/debug_plots', exist_ok=True)

    if config.analysis.run_heavy_edge_metrics:

        f = open(config.save.folder + "/pred_list.txt", 'r')
        pred_list = f.readlines()
        f.close()

        f = open(config.analysis.edge_image_list, 'r')
        gt_list = f.readlines()
        f.close()

        precision_vec, recall_vec = pr_evaluation(gt_list, pred_list, save_folder=config.save.folder)
        precision_recall_vec = np.vstack((precision_vec, recall_vec)).transpose()

        bsds_edge_cols = ['Precision', 'Recall']
        bsds_edge_mat = [precision_vec, recall_vec]
        bsds_edge_mat = list(map(list, zip(*bsds_edge_mat)))
        bsds_edge_frm_metrics = pd.DataFrame(bsds_edge_mat, columns=bsds_edge_cols)
        bsds_edge_frm_metrics.to_csv(config.save.folder + '/sfm_analysis/debug_plots/mean_frames_bsds_edge_metrics.csv')

        plot_edge_graph(precision_vec, recall_vec,
                        None, None,
                        config.save.folder + '/sfm_analysis/debug_plots/frames_bsds_edge_metrics_plot.png')

        plot_edge_graph(precision_vec, recall_vec,
                        None, None,
                        config.save.folder + '/sfm_analysis/debug_plots/frames_bsds_edge_metrics_plot_fixed_ax.png',
                        fixed_lims=True)

        # AUC over all range
        f1 = mean_recall_at_precision_range(precision_recall_vec)
        # AUC over large intersection range (see results in paper)
        f2 = mean_recall_at_precision_range(precision_recall_vec, small_lim=config.analysis.prec_recall_eval_range_min,
                                            large_lim=config.analysis.prec_recall_eval_range_max)

        print('AUC over all range: ' + str(f1))
        print('AUC over partial range: ' + str(f2))

        f = open(config.save.folder + '/sfm_analysis/debug_plots/edge_AUC.txt', 'w')
        f.writelines('AUC over all range: ' + str(f1) + '\n')
        f.writelines('AUC over partial range: ' + str(f2) + '\n')
        f.close()

    if config.analysis.run_metrics:

        frm_metrics = run_depth_metrics(config.save.folder + "/pred_list.txt",
                                        config.analysis.gt_image_list,
                                        config.save.folder + "/pred_list.txt",
                                        config.save.folder + '/sfm_analysis/debug_plots',
                                        config)
        frm_metrics.to_csv(config.save.folder + '/sfm_analysis/debug_plots/frames_depth_metrics.csv')
        frm_metrics_mean = frm_metrics.mean()
        frm_metrics_mean.to_csv(config.save.folder + '/sfm_analysis/debug_plots/mean_frames_depth_metrics.csv')

        ord_error = run_ord_metrics(config.analysis.gt_image_list,
                                    config.save.folder + "/pred_list.txt")
        ord_error_mean = ord_error.mean()
        f = open(config.save.folder + '/sfm_analysis/debug_plots/mean_frames_ord_metrics.txt', 'w')
        f.writelines(str(ord_error_mean))
        f.close()


def run_depth_metrics(image_list_path, gt_image_list_path, depth_pred_list_path, output_dir, cfg):
    # out_dir_path = output_dir + '/sfm_analysis'
    # if not os.path.exists(out_dir_path):
    #     os.makedirs(out_dir_path)
    #
    # vis_out_dir_path = output_dir + '/sfm_analysis/debug_plots'
    # if not os.path.exists(vis_out_dir_path):
    #     os.makedirs(vis_out_dir_path)

    data_loader = DataLoader(
        image_list_path=image_list_path,
        gt_list_path=gt_image_list_path,
        depth_pred_list_path=depth_pred_list_path
    )
    print('Loaded {} test image paths.'.format(len(data_loader)))

    # TODO: Remove SparsePredictionAnalyzer.
    analyzer = DensePredictionAnalyzer(analysis_cfg=cfg.analysis)

    run_analysis(data_loader, analyzer, cfg)

    frm_metrics = pd.DataFrame(analyzer.per_frm_res, columns=analyzer.columns)

    return frm_metrics

def plot_edge_graph(chamfer_per_vec_mean_1, chamfer_per_vec_mean_2, chamfer_per_vec_mean_lidar_1=None,
                    chamfer_per_vec_mean_lidar_2=None, save_file_path='', fixed_lims=False):
    plt.plot(chamfer_per_vec_mean_1, chamfer_per_vec_mean_2, 'bo-', label='RGB only')
    if not chamfer_per_vec_mean_lidar_1 is None:
        plt.plot(chamfer_per_vec_mean_lidar_1, chamfer_per_vec_mean_lidar_2, 'ro-', label='LIDAR input')
    plt.title('Edge precision to recall (approximate edge metric)')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend()
    if fixed_lims:
        plt.xlim([0.1, 1])
        plt.ylim([0.1, 1])
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    # plt.show()
    fig.savefig(save_file_path)
    plt.close(fig)


@torch.no_grad()
def infer_and_save_depth(input_file, lidar_input_file, rgb_edge_input_file, output_file, model_wrapper, config,
                         image_shape, crop_shape, half, save, counter):
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
        os.makedirs(output_file, exist_ok=True)
        output_dir_path = output_file
        # os.path.basename(input_file)
        output_file = os.path.join(output_file, str.zfill(str(counter), 8) + '.png')

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file)
    original_shape = image.size
    # Resize and to tensor
    # image_shape = image.size
    if not len(image_shape) == 0:
        image = resize_image(image, image_shape)
    if len(crop_shape) == 2:
        # image.show()
        im_size = image.size
        start_x = int((im_size[0]-crop_shape[1])/2)
        start_y = int((im_size[1]-crop_shape[0]))
        image = image.crop((start_x, start_y, start_x+crop_shape[1], start_y+crop_shape[0]))
        # print('Done')

    image = to_tensor(image).unsqueeze(0)

    if lidar_input_file.split('.')[-1] == 'png':
        lidar_image = read_png_depth(lidar_input_file)
        lidar_image[lidar_image < 0.0] = 0.0
    elif lidar_input_file.split('.')[-1] == 'npz':
        lidar_image = read_npz_depth(lidar_input_file, 'velodyne')
        lidar_image[lidar_image < 0.0] = 0.0
    elif lidar_input_file.split('.')[-1] == 'bin':

        if config.datasets.test.dataset[0] == 'KITTI':
            lidar_rows = np.fromfile(lidar_input_file, dtype=np.float32)
            lidar_rows = lidar_rows.reshape((-1, 4))
            lidar_rows = lidar_rows.astype('int')

            lidar_image = np.zeros(original_shape)
            lidar_image[lidar_rows[:, 1], lidar_rows[:, 0]] = lidar_rows[:, 2]
        elif config.datasets.test.dataset[0] == 'GTA':
            # only for GTA
            K = np.array([960, 0, 960, 0, 960, 540, 0, 0, 1]).reshape([3, 3])
            raw_lidar_map = read_lidar(lidar_input_file)
            lidar_image = process_lidar(raw_lidar_map, K)
    else:
        lidar_image = None

    if rgb_edge_input_file is not None:
        if rgb_edge_input_file.split('.')[-1] == 'npy':
            rgb_edge_image = np.load(rgb_edge_input_file)
        elif rgb_edge_input_file.split('.')[-1] == 'png':
            rgb_edge_image = cv2.imread(rgb_edge_input_file)[:, :, 0]
            rgb_edge_image = rgb_edge_image / 255
        rgb_edge_image = to_tensor(rgb_edge_image).unsqueeze(0)
        rgb_edge_image = rgb_edge_image.to('cuda:{}'.format(rank()), dtype=dtype)
    else:
        rgb_edge_image = None

    # if not lidar_image is None:
    if not config.datasets.test.input_depth_type[0] == '':
        lidar_image = resize_depth_preserve(lidar_image, image_shape)
        lidar_image = to_tensor(lidar_image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)
        if not config.datasets.test.input_depth_type[0] == '':
            lidar_image = lidar_image.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image, rgb_edge=rgb_edge_image)['inv_depths'][0][0]

    pred_depth = inv2depth(pred_inv_depth)

    if not config.datasets.test.input_depth_type[0] == '':
        pred_inv_depth_rgbd = model_wrapper.depth(image, lidar_image, rgb_edge=rgb_edge_image)['inv_depths'][0][0]
        pred_depth_rgbd = inv2depth(pred_inv_depth_rgbd)

    # Prepare RGB image
    rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    # Prepare inverse depth
    viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
    image = viz_pred_inv_depth
    # Save visualization
    print('Saving {} to {}'.format(
        pcolor(input_file, 'cyan', attrs=['bold']),
        pcolor(output_file, 'magenta', attrs=['bold'])))
    # imwrite(output_file, image[:, :, ::-1])
    pred_numpy = pred_depth[0, 0, :, :].detach().cpu().numpy()
    pred_max = pred_numpy.max()
    imwrite(output_file[:-4] + '_regular.png', (pred_numpy / pred_max) * 255)
    if config.save.depth.npz:
        np.save(output_file[:-4] + '_regular.npy', pred_numpy)

    depth_im_color = np.log(pred_numpy)

    depth_im_color = depth_im_color - np.min(depth_im_color)
    depth_im_color = depth_im_color / np.max(depth_im_color)
    depth_log = depth_im_color

    vmax = np.percentile(depth_log, 100)
    normalizer = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=plt.get_cmap('Spectral'))
    colormapped_im = (mapper.to_rgba(depth_log)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    im.save(output_file[:-4] + '_regular_color.png')


def run_ord_metrics(gt_image_list_path, depth_pred_list_path):

    f_pred = open(depth_pred_list_path, 'r')
    depth_pred_list = f_pred.readlines()
    f_gt = open(gt_image_list_path, 'r')
    gt_list = f_gt.readlines()

    pairs_for_rand = [5000, 2500, 1000, 500, 100]

    ord_error = np.zeros(len(depth_pred_list))

    for i in range(0, len(depth_pred_list)):

        pred_im = np.load(depth_pred_list[i].split('\n')[0])

        gt_im = cv2.imread(gt_list[i].split('\n')[0])[:, :, 0]

        gt_im = resize_depth_preserve(gt_im, pred_im.shape[0:2])[:, :, 0]

        centers = np.where(gt_im > 0)

        pairs_index = 0
        cur_pairs_for_rand = pairs_for_rand[pairs_index]
        while len(centers[0]) < int(cur_pairs_for_rand*2):
            pairs_index = pairs_index + 1
            cur_pairs_for_rand = pairs_for_rand[pairs_index]

        rand_perm = np.random.permutation(len(centers[0]))[:2*cur_pairs_for_rand]

        gt_pairs = np.reshape(rand_perm, (int(len(rand_perm)/2),2))

        ord_ratio = d3r(gt_im, pred_im, centers, gt_pairs)
        ord_error[i]=1-(np.sum(ord_ratio)/len(ord_ratio))

    return ord_error


if __name__ == '__main__':
    main(sys.argv[1:])
    # main(sys.argv[1:])

