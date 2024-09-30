# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC). All rights reserved.
# Adpated from: https://github.com/Britefury/py-bsds500

from collections import namedtuple
import numpy as np
from bsds_metric.bsds import thin, correspond_pixels
import multiprocessing
import cv2
import glob
import pickle
import pandas as pd
from edge import edge_from_depth
import os
import argparse
from PIL import Image

def evaluate_boundaries_bin(predicted_boundaries_bin, gt_boundaries,
                            max_dist=0.0075, apply_thinning=True):
    """
    Evaluate the accuracy of a predicted boundary.

    :param predicted_boundaries_bin: the predicted boundaries as a (H,W)
    binary array
    :param gt_boundaries: a list of ground truth boundaries, as returned
    by the `load_boundaries` or `boundaries` methods
    :param max_dist: (default=0.0075) maximum distance parameter
    used for determining pixel matches. This value is multiplied by the
    length of the diagonal of the image to get the threshold used
    for matching pixels.
    :param apply_thinning: (default=True) if True, apply morphologial
    thinning to the predicted boundaries before evaluation
    :return: tuple `(count_r, sum_r, count_p, sum_p)` where each of
    the four entries are float values that can be used to compute
    recall and precision with:
    ```
    recall = count_r / (sum_r + (sum_r == 0))
    precision = count_p / (sum_p + (sum_p == 0))
    ```
    """
    acc_prec = np.zeros(predicted_boundaries_bin.shape, dtype=bool)
    predicted_boundaries_bin = predicted_boundaries_bin != 0

    if apply_thinning:
        predicted_boundaries_bin = thin.binary_thin(predicted_boundaries_bin)

    sum_r = 0
    count_r = 0
    for gt in gt_boundaries:
        match1, match2, cost, oc = correspond_pixels.correspond_pixels(
            predicted_boundaries_bin, gt, max_dist=max_dist
        )
        match1 = match1 > 0
        match2 = match2 > 0
        # Precision accumulator
        acc_prec = acc_prec | match1
        # Recall
        sum_r += gt.sum()
        count_r += match2.sum()

    # Precision
    sum_p = predicted_boundaries_bin.sum()
    count_p = acc_prec.sum()

    return count_r, sum_r, count_p, sum_p

def evaluate_boundaries(predicted_boundaries, gt_boundaries,
                        thresholds=99, max_dist=0.0075, apply_thinning=True,
                        progress=None):
    """
    Evaluate the accuracy of a predicted boundary and a range of thresholds

    :param predicted_boundaries: the predicted boundaries as a (H,W)
    floating point array where each pixel represents the strength of the
    predicted boundary
    :param gt_boundaries: a list of ground truth boundaries, as returned
    by the `load_boundaries` or `boundaries` methods
    :param thresholds: either an integer specifying the number of thresholds
    to use or a 1D array specifying the thresholds
    :param max_dist: (default=0.0075) maximum distance parameter
    used for determining pixel matches. This value is multiplied by the
    length of the diagonal of the image to get the threshold used
    for matching pixels.
    :param apply_thinning: (default=True) if True, apply morphologial
    thinning to the predicted boundaries before evaluation
    :param progress: a function that can be used to monitor progress;
    use `tqdm.tqdm` or `tdqm.tqdm_notebook` from the `tqdm` package
    to generate a progress bar.
    :return: tuple `(count_r, sum_r, count_p, sum_p, thresholds)` where each
    of the first four entries are arrays that can be used to compute
    recall and precision at each threshold with:
    ```
    recall = count_r / (sum_r + (sum_r == 0))
    precision = count_p / (sum_p + (sum_p == 0))
    ```
    The thresholds are also returned.
    """
    if progress is None:
        progress = lambda x, *args, **kwargs: x

    # Handle thresholds
    if isinstance(thresholds, int):
        thresholds = np.linspace(1.0 / (thresholds + 1),
                                 1.0 - 1.0 / (thresholds + 1), thresholds)
    elif isinstance(thresholds, np.ndarray):
        if thresholds.ndim != 1:
            raise ValueError('thresholds array should have 1 dimension, '
                             'not {}'.format(thresholds.ndim))
        pass
    else:
        raise ValueError('thresholds should be an int or a NumPy array, not '
                         'a {}'.format(type(thresholds)))

    sum_p = np.zeros(thresholds.shape)
    count_p = np.zeros(thresholds.shape)
    sum_r = np.zeros(thresholds.shape)
    count_r = np.zeros(thresholds.shape)

    for i_t, thresh in enumerate(progress(list(thresholds))):
        predicted_boundaries_bin = predicted_boundaries >= thresh

        acc_prec = np.zeros(predicted_boundaries_bin.shape, dtype=bool)

        if apply_thinning:
            predicted_boundaries_bin = thin.binary_thin(
                predicted_boundaries_bin)

        for gt in gt_boundaries:

            match1, match2, cost, oc = correspond_pixels.correspond_pixels(
                predicted_boundaries_bin, gt, max_dist=max_dist
            )
            match1 = match1 > 0
            match2 = match2 > 0
            # Precision accumulator
            acc_prec = acc_prec | match1
            # Recall
            sum_r[i_t] += gt.sum()
            count_r[i_t] += match2.sum()

        # Precision
        sum_p[i_t] = predicted_boundaries_bin.sum()
        count_p[i_t] = acc_prec.sum()

    return count_r, sum_r, count_p, sum_p, thresholds

def compute_rec_prec_f1(count_r, sum_r, count_p, sum_p):
    """
    Computer recall, precision and F1-score given `count_r`, `sum_r`,
    `count_p` and `sum_p`; see `evaluate_boundaries`.
    :param count_r:
    :param sum_r:
    :param count_p:
    :param sum_p:
    :return: tuple `(recall, precision, f1)`
    """
    rec = count_r / (sum_r + (sum_r == 0))
    prec = count_p / (sum_p + (sum_p == 0))
    f1_denom = (prec + rec + ((prec+rec) == 0))
    f1 = 2.0 * prec * rec / f1_denom
    return rec, prec, f1


SampleResult = namedtuple('SampleResult', ['sample_name', 'threshold',
                                           'recall', 'precision', 'f1'])
ThresholdResult = namedtuple('ThresholdResult', ['threshold', 'recall',
                                                 'precision', 'f1'])
OverallResult = namedtuple('OverallResult', ['threshold', 'recall',
                                             'precision', 'f1',
                                             'best_recall', 'best_precision',
                                             'best_f1', 'area_pr'])

EvalResult = namedtuple('EvalResult', ['count_r_overall', 'sum_r_overall',
                                          'count_p_overall', 'sum_p_overall',
                                          'count_r_best', 'sum_r_best',
                                          'count_p_best', 'sum_p_best',
                                          'used_thresholds', 'recall', 'precision'])

def _pred_eval(pred_path, gt_path, crop):
    # Get the paths for the ground truth and predicted boundaries

    if os.path.exists(crop.split('\n')[0]):
        is_image_crop = True
        crop = cv2.imread(crop.split('\n')[0])
        crop = crop[:,:,0]/255
        print(str(crop.shape))
    else:
        is_image_crop = False
        crop = eval(crop)

    pred = cv2.imread(pred_path.split('\n')[0])
    pred = pred[:, :, 0] / 255
    pred[pred > 0.5] = 1.0
    pred[pred < 0.5] = 0.0
    if not is_image_crop:
        if len(crop) > 0:
            pred = pred[crop[2]:crop[3], crop[0]:crop[1]]
    else:
        pred = pred*crop
        print(str(pred.shape))

    gt_b = cv2.imread(gt_path.split('\n')[0])
    gt_b = gt_b[:, :, 0] / 255
    gt_b[gt_b > 0.5] = 1.0
    gt_b[gt_b < 0.5] = 0.0
    if not is_image_crop:
        if len(crop) > 0:
            gt_b = gt_b[crop[2]:crop[3], crop[0]:crop[1]]
    else:
        gt_b = gt_b*crop
    # Evaluate predictions
    count_r, sum_r, count_p, sum_p, used_thresholds = \
        evaluate_boundaries(pred, [gt_b], thresholds=1,
                            apply_thinning=False,
                            max_dist=0.002)

    # Compute precision, recall and F1
    rec, prec, f1 = compute_rec_prec_f1(count_r, sum_r, count_p, sum_p)

    # Find best F1 score
    best_ndx = np.argmax(f1)

    count_r_best = count_r[best_ndx]
    sum_r_best = sum_r[best_ndx]
    count_p_best = count_p[best_ndx]
    sum_p_best = sum_p[best_ndx]

    return EvalResult(count_r, sum_r, count_p, sum_p,
                      count_r_best, sum_r_best, count_p_best, sum_p_best,
                      used_thresholds, rec, prec)

def pr_evaluation(edge_list,
                  pred_list,
                  edge_thresh_range=None,
                  gt_crop=[44, 1197, 153, 371],
                  min_depth=0.0,
                  max_depth=80.0,
                  save_folder='temp_output',
                  num_workers=4):

    os.makedirs(save_folder, exist_ok=True)

    if edge_thresh_range is None:
        edge_thresh_range = list(range(20, 241, 20))

    precision_vec = []
    recall_vec = []

    # f_pred = open(pred_list, 'r')
    # depth_pred_list = f_pred.readlines()
    # f_gt = open(edge_list, 'r')
    # edge_gt_list = f_gt.readlines()

    depth_pred_list = pred_list
    edge_gt_list = edge_list

    # LT: if the edge gt list is multiscale, take only the first entry
    if len(edge_gt_list) > len(depth_pred_list):
        ratio = len(edge_gt_list) / len(depth_pred_list)
        edge_gt_list = edge_gt_list[0:len(edge_gt_list):int(ratio)]

    pool = multiprocessing.Pool(num_workers)

    for thresh_val in edge_thresh_range:

        print('BSDS thresh: ' + str(thresh_val))

        for i in range(0,len(depth_pred_list)):

            edge_gt_im = cv2.imread(edge_gt_list[i].split('\n')[0])[:,:,0]
            name_pred_edge_im = os.path.join(save_folder, "{:010d}_pred_canny_edge.jpeg".format(i))

            edge_pred_im = edge_from_depth(depth_pred_list[i],
                                           (edge_gt_im.shape[1], edge_gt_im.shape[0]),
                                           # (1253,360),
                                           name_pred_edge_im,
                                           min_depth=min_depth,
                                           max_depth=max_depth,
                                           thresh_1=int(thresh_val / 2),
                                           thresh_2=int(thresh_val))



        pred_path_for_heavy_edge_metrics = os.path.join(save_folder, "*_pred_canny_edge.jpeg")
        pred_list = sorted(glob.glob(pred_path_for_heavy_edge_metrics))

        gt_list = edge_gt_list

        sample_results = []

        # num_images = len(gt_list)

        crop = gt_crop
        crop_list = [str(crop)]*len(gt_list)

        eval_arr = pool.starmap(_pred_eval, zip(pred_list, gt_list, crop_list))

        count_r_overall = sum([x[0] for x in eval_arr])
        sum_r_overall = sum([x[1] for x in eval_arr])
        count_p_overall = sum([x[2] for x in eval_arr])
        sum_p_overall = sum([x[3] for x in eval_arr])
        count_r_best = sum([x[4] for x in eval_arr])
        sum_r_best = sum([x[5] for x in eval_arr])
        count_p_best = sum([x[6] for x in eval_arr])
        sum_p_best = sum([x[7] for x in eval_arr])
        used_thresholds = [x[8] for x in eval_arr][0]
        total_precision = [x[10][0] for x in eval_arr]
        total_recall = [x[9][0] for x in eval_arr]

        # Computer overall precision, recall and F1
        rec_overall, prec_overall, f1_overall = compute_rec_prec_f1(
            count_r_overall, sum_r_overall, count_p_overall, sum_p_overall)

        # Find best F1 score
        best_i_ovr = np.argmax(f1_overall)

        threshold_results = []
        for thresh_i in range(1):
            threshold_results.append(ThresholdResult(used_thresholds[thresh_i],
                                                     rec_overall[thresh_i],
                                                     prec_overall[thresh_i],
                                                     f1_overall[thresh_i]))

        rec_unique, rec_unique_ndx = np.unique(rec_overall, return_index=True)
        prec_unique = prec_overall[rec_unique_ndx]
        if rec_unique.shape[0] > 1:
            prec_interp = np.interp(np.arange(0, 1, 0.01), rec_unique,
                                    prec_unique, left=0.0, right=0.0)
            area_pr = prec_interp.sum() * 0.01
        else:
            area_pr = 0.0

        rec_best, prec_best, f1_best = compute_rec_prec_f1(
            float(count_r_best), float(sum_r_best), float(count_p_best),
            float(sum_p_best)
        )

        overall_result = OverallResult(used_thresholds[best_i_ovr],
                                       rec_overall[best_i_ovr],
                                       prec_overall[best_i_ovr],
                                       f1_overall[best_i_ovr], rec_best,
                                       prec_best, f1_best, area_pr)

        precision_vec.append(overall_result[2])
        recall_vec.append(overall_result[1])


    return precision_vec, recall_vec



def load_depth_edges_gt_from_txt(filepath, H, W):

    img = Image.new('1', (W, H), 0)  # Create a new black image
    pixels = img.load()

    with open(filepath, 'r') as file:
        for line in file:
            y, x = map(int, line.strip().split(' '))
            if 0 <= x < W and 0 <= y < H:
                pixels[x, y] = 1  # Set pixel to white

    return pixels

def mean_recall_at_precision_range(arr, small_lim=0.0, large_lim=1.0):

    interp_x = np.array(range(int(small_lim*100),int(large_lim*100)))/100
    interp_y = np.interp(interp_x, arr[:, 0], arr[:, 1])

    interp_y[interp_y < 0] = 0
    interp_y[interp_y > 1] = 1

    mean_recall = np.mean(interp_y)

    return mean_recall

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate images from coordinate text files.')
    parser.add_argument('--depth_pred_list_path', type=str,
                        help='List of predicted depth image *names* in .npy format (with metric depth)')
    parser.add_argument('--depth_pred_dir_path', type=str,
                        help='Path to the directory that contains the depth images (.npy files)')
    parser.add_argument('--depth_edge_gt_list_path', default='data/kitti_de/kitti_de_annotated_edges.txt', type=str,
                        help='List of GT depth edgeimage *names* in .png format (with metric depth)')
    parser.add_argument('--depth_edge_gt_dir_path', default='data/kitti_de/gt', type=str,
                        help='Path to the directory that contains the depth edges GT (.png files)')
    parser.add_argument('--temp_save_path', default='temp_output', type=str,
                        help='Temp directory for the depth edges images per thresh')
    parser.add_argument('--prec_recall_eval_range_min', default=0.12, type=float,
                        help='Min range for edge AUC computation')
    parser.add_argument('--prec_recall_eval_range_max', default=0.65, type=float,
                        help='Min range for edge AUC computation')

    args = parser.parse_args()

    f = open(args.depth_pred_list_path, 'r')
    pred_list = f.readlines()
    f.close()
    pred_list = [args.depth_pred_dir_path + '/' + x.split('\n')[0].split('/')[-1] for x in pred_list]

    f = open(args.depth_edge_gt_list_path, 'r')
    gt_list = f.readlines()
    f.close()
    gt_list = [args.depth_edge_gt_dir_path + '/' + x.split('\n')[0].split('/')[-1] for x in gt_list]

    precision_vec, recall_vec = pr_evaluation(gt_list, pred_list, save_folder=args.temp_save_path)
    precision_recall_vec = np.vstack((precision_vec, recall_vec)).transpose()

    # AUC over all range
    f1 = mean_recall_at_precision_range(precision_recall_vec)
    # AUC over large intersection range (see results in paper)
    f2 = mean_recall_at_precision_range(precision_recall_vec, small_lim=args.prec_recall_eval_range_min, large_lim=args.prec_recall_eval_range_max)

    print('AUC over all range: ' + str(f1) + '\n')
    print('AUC over partial range: ' + str(f2) + '\n')