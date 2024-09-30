# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC). All rights reserved.

import os
import pickle
from PIL import Image
import cv2
import numpy as np
import pandas as pd

from packnet_code.packnet_sfm.datasets.augmentations import resize_depth_preserve


def depth_read(filename):

    depth_png = np.array(Image.open(filename), dtype=int)
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.

    return depth


def depth_read_npy(filename):
    depth = np.load(filename)
    depth[depth == 0] = -1.
    return depth

def ndcToDepth_vec(ndc):
    nc_z = 0.15
    fc_z = 600

    rows, cols = ndc.shape

    nc_z_mat = np.ones((rows, cols)) * nc_z
    d_nc = nc_z_mat

    depth = d_nc / (ndc + (nc_z_mat * d_nc / (2 * fc_z)))
    depth[ndc == 0.0] = fc_z

    return depth

def read_lidar(data_array):

    xyzi = data_array.reshape(-1, 4)

    x = xyzi[:, 0]
    y = xyzi[:, 1]
    z = xyzi[:, 2]
    i = xyzi[:, 3]

    all_points = np.vstack((-y, -z, x)).T

    # Remove nan points
    nan_mask = ~np.any(np.isnan(all_points), axis=1)
    point_cloud = all_points[nan_mask].T

    return point_cloud


def depth_read_bin(filename, rows=1080, cols=1920):
    # try to read as full res depth images, if fails, read as sparse lidar
    fd = open(filename.split('\n')[0], 'rb')
    f = np.fromfile(fd, dtype=np.float32, count=rows*cols)
    # full GT
    if f.size == rows*cols:
        ndc = f.reshape((rows, cols))
        depth = ndcToDepth_vec(ndc)
        return depth
    # lidar
    else:
        K = np.array([960, 0, 960, 0, 960, 540, 0, 0, 1]).reshape([3, 3])
        raw_lidar_map = read_lidar(f)

        lidar_mat = np.zeros((1080, 1920))
        # lidar_dim, num_lidar_pts = raw_lidar_map.shape
        # vio_count = 0
        p = np.matmul(K, raw_lidar_map)
        p_norm = p / p[2, :]
        in_range_idx = np.logical_and(
            np.logical_and(np.logical_and(p_norm[0, :] >= 0, p_norm[0, :] < 1920), p_norm[1, :] >= 0),
            p_norm[1, :] < 1080)
        p_norm = p_norm[:, in_range_idx].astype('int')
        p = p[:, in_range_idx]
        lidar_mat[p_norm[1, :], p_norm[0, :]] = p[2, :]

        return lidar_mat



def disp_to_depth(disp, min_depth, max_depth):

    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    scaled_disp[np.equal(disp, -1)] = -1
    depth[np.equal(disp, -1)] = -1

    return scaled_disp, depth


def load_data(image_path, gt_image_path, depth_pred_path, gt_type="mono"):
    img = cv2.imread(image_path, 1)
    gt = depth_read(gt_image_path)
    d = np.load(depth_pred_path)

    if gt_type == "stereo":
        _, gt = disp_to_depth(gt, 0.1, 100)

    return img, gt, d


def dense_to_gt_match(gt, d):
    d_filt = np.copy(d)
    d_gt_med = np.median(gt[np.where(gt != -1)])
    d_filt_med = np.median(d[np.where(gt != -1)])
    d_filt = d_filt * d_gt_med / d_filt_med
    # d_filt[np.where(d_filt > 80)] = 80
    d_filt[np.where(gt == -1)] = -1
    return d_filt


def get_gt_bins(gt, bin_num):
    d_gt_uni = np.unique(gt)
    gt_bins = np.linspace(0, d_gt_uni.max() ,bin_num).tolist()
    gt_bins += [np.inf]
    return gt_bins


def nanify(gt, d):
    d[np.where(gt == -1)] = np.nan
    gt[np.where(gt == -1)] = np.nan
    return gt, d


class DataLoader:
    def __init__(self,
                 image_list_path,
                 gt_list_path,
                 depth_pred_list_path,
                 ):
        self.image_list_path = image_list_path
        self.gt_list_path = gt_list_path
        self.depth_pred_list_path = depth_pred_list_path

        self.path_lists = None
        self.relative_poses = None
        self.intrinsics = None
        self.num_images = None

        self._read_lists()


    def _read_lists(self):
        image_list = open(self.image_list_path).read().splitlines()
        gt_list = open(self.gt_list_path).read().splitlines()
        depth_pred_list = open(self.depth_pred_list_path).read().splitlines()

        self.path_lists = {
            "image": image_list,
            "depth_gt": gt_list,
            "depth_pred": depth_pred_list,
        }

        self.num_images = len(image_list)
        self._check_data()

    def __getitem__(self, frm_idx):
        assert self.num_images is not None
        assert (frm_idx >= 0) and (frm_idx < self.num_images)

        im_path = self.path_lists["image"][frm_idx]
        d_gt_path = self.path_lists["depth_gt"][frm_idx]
        d_path = self.path_lists["depth_pred"][frm_idx]

        im = cv2.imread(im_path)
        d_gt = self._load_gt(d_gt_path)
        d = self._load_depth_pred(d_path)
        W, H = d.shape[:2]

        # Resize prediction to fit GT.
        if d.shape != d_gt.shape:
            d_gt = resize_depth_preserve(d_gt, (W, H))[:,:,0]

        return im, d_gt, d

    def __len__(self):
        return self.num_images

    def _load_gt(self, gt_image_path):
        if gt_image_path.endswith('.npy'):
            gt = depth_read_npy(gt_image_path)
        elif gt_image_path.endswith('.png'):
            gt = depth_read(gt_image_path)
        elif gt_image_path.endswith('.bin'):
            gt = depth_read_bin(gt_image_path)
        else:
            raise ValueError("Depth GT must be in .png or .npy format.")
        return gt

    def _load_depth_pred(self, depth_pred_path):
        depth_pred = None

        # This can be an image or table.
        if depth_pred_path.endswith(".npy"):
            depth_pred = np.load(depth_pred_path)

        # Assume that if txt we have tabular data.
        if depth_pred_path.endswith(".txt"):
            depth_pred = np.genfromtxt(depth_pred_path, delimiter=',')

        assert depth_pred is not None
        return depth_pred

    def _check_data(self):
        assert len(self.path_lists["image"]) == self.num_images
        assert len(self.path_lists["depth_gt"]) == self.num_images
        assert len(self.path_lists["depth_pred"]) == self.num_images



class Analyzer:
    def __init__(self, analysis_cfg):
        self.cfg = analysis_cfg

        # Columns for DataFrame.
        self.columns = [
            "frm_idx",
            "mean_rel_err",
            "std_rel_err",
            "abs_rel_err",
            "accuracy_1p1",
            "accuracy_1p25",
            "median_scale_factor"
        ]

        # self.class_columns = [
        #     "frm_idx",
        #     "mean_rel_err",
        #     "std_rel_err",
        #     "accuracy_1p25",
        #     "class_accuracy_1p1",
        #     "class_accuracy_1p25",
        #     "class_rel_err",
        #     "class_mean_rel_err",
        #     "class_std_rel_err"
        # ]

        self.per_frm_res             = []
        # self.bin_edges               = None
        # self.histograms              = []
        # self.acc_vs_depth_1p1        = []
        # self.acc_vs_depth_1p25       = []
        # self.abs_rel_err_vs_conf     = []

        # self.pts_vs_depth            = []
        # self.depth_bin_edges         = None
        #
        # self.class_histograms        = []
        # self.class_per_frm_res       = []
        # self.class_acc_vs_depth_1p25 = []
        # self.class_acc_vs_depth_1p1  = []
        # self.class_pts_vs_depth      = []
        #
        # self.rel_trans               = []
        # self.rel_rot                 = []


    def eval_metrics(self):
        raise NotImplementedError

    def _match_scales_using_medians(self):
        raise NotImplementedError

    def calc_rel_err_histogram(self):
        raise NotImplementedError

    def calc_acc_vs_depth_histogram(self):
        raise NotImplementedError

    def store_frm_results(self, frm_idx, frm_metrics):
        self.per_frm_res.append(
            [
                frm_idx,
                frm_metrics["vals"]["mean_rel_err"],
                frm_metrics["vals"]["std_rel_err"],
                frm_metrics["vals"]["abs_rel_err"],
                frm_metrics["vals"]["accuracy_1p1"],
                frm_metrics["vals"]["accuracy_1p25"],
                frm_metrics["vals"]["median_scale_factor"],
            ]
        )
        # self.class_per_frm_res.append(
        #     [
        #         frm_idx,
        #         frm_metrics["vals"]["mean_rel_err"],
        #         frm_metrics["vals"]["std_rel_err"],
        #         frm_metrics["vals"]["accuracy_1p25"]
        #     ]
        # )

    def get_frm_metrics_df(self):
        return pd.DataFrame(self.per_frm_res, columns=self.columns)

    # def class_get_frm_metrics_df(self):
    #     return pd.DataFrame(self.class_per_frm_res, columns=self.class_columns)

    def save_results(self, output_dir, out_file_name, verbose=True):
        if verbose:
            print("Saving analyzer data ...")

        pickle_path = os.path.join(output_dir, out_file_name)

        results = {
            "columns"                      : self.columns,
            # "class_columns"                : self.class_columns,

            "per_frm_res"                  : np.array(self.per_frm_res),
            # "class_per_frm_res"            : np.array(self.class_per_frm_res),

            "analysis_cfg"                 : dict(self.cfg),
        }

        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)

        if verbose:
            print("Done saving analyzer data.")


class DensePredictionAnalyzer(Analyzer):
    def __init__(self, analysis_cfg):
        super(DensePredictionAnalyzer, self).__init__(analysis_cfg)

        self.cfg = analysis_cfg

        if not analysis_cfg.eval_mask_image_list == '':
            f = open(analysis_cfg.eval_mask_image_list)
            eval_mask_list = f.readlines()
            f.close()
            self.eval_mask_list = eval_mask_list
        else:
            self.eval_mask_list = None

    # def eval_frame(self, frm_idx, gt, d, pose, K, conf):
    def eval_frame(self, frm_idx, gt, d, gt_crop_im=None):
        """Calculate evaluation metrics."""

        gt = self._process_depth_gt_for_evaluation(
            gt, self.cfg.min_depth, self.cfg.max_depth, self.cfg.gt_crop, gt_crop_im
        )

        scale_factor = 1

        frm_metrics = self.calc_metrics(gt, d, scale_factor)

        self.store_frm_results(frm_idx, frm_metrics)

        return frm_metrics

    def calc_metrics(self, gt, d, scale_factor):

        frm_metrics = {'maps': {}, 'vals': {}, 'hists': {}, 'poses': {}, 'curves': {}}

        self._calc_full_frame_metrics(frm_metrics, gt, d, scale_factor)

        return frm_metrics

    # def _calc_full_frame_metrics(self, frm_metrics, gt, d, scale_factor, pose, conf):
    def _calc_full_frame_metrics(self, frm_metrics, gt, d, scale_factor):
        # Images.
        frm_metrics['maps']['rel_err']       = rel_err(d, gt)
        frm_metrics['maps']['abs_rel_err']   = abs_rel_err(d, gt)
        frm_metrics['maps']['accuracy_1p1']  = accuracy(d, gt, 1.1)
        frm_metrics['maps']['accuracy_1p25'] = accuracy(d, gt, 1.25)
        frm_metrics['maps']['d_med']         = d

        # Scalars.
        frm_metrics['vals']['abs_rel_err']         = np.nanmean(abs_rel_err(d, gt))
        frm_metrics['vals']['accuracy_1p1']        = np.nanmean(accuracy(d, gt, 1.1))
        frm_metrics['vals']['accuracy_1p25']       = np.nanmean(accuracy(d, gt, 1.25))
        frm_metrics["vals"]["median_scale_factor"] = scale_factor
        frm_metrics['vals']['mean_rel_err']        = np.nanmean(frm_metrics['maps']['rel_err'])
        frm_metrics['vals']['std_rel_err']         = np.nanstd(frm_metrics['maps']['rel_err'])


    def _process_depth_gt_for_evaluation(self, gt, min_depth, max_depth, gt_crop, gt_crop_im=None):

        H, W = gt.shape
        mask = np.logical_and(gt > min_depth, gt < max_depth)

        if gt_crop_im is None:
            crop = np.array([gt_crop[2], gt_crop[3],
                             gt_crop[0], gt_crop[1]])
            crop = crop.astype(np.int32)

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        else:
            crop_mask = gt_crop_im > 0

        mask = np.logical_and(mask, crop_mask)


        gt_out = gt.copy()
        gt_out[np.logical_not(mask)] = -1

        return gt_out

    def _process_depth_prediction_for_evaluation(self, depth_pred):
        return depth_pred


def mse(d, d_gt):
    metric = (d - d_gt) ** 2
    metric[np.where(d_gt == -1)] = np.nan
    return metric


def mse_log(d, d_gt):
    d_log = np.copy(d)
    d_gt_log = np.copy(d_gt)
    d_log[np.where(d != -1)] = np.log10(d[np.where(d != -1)])
    d_gt_log[np.where(d_gt != -1)] = np.log10(d_gt[np.where(d_gt != -1)])
    metric = (d_log - d_gt_log) ** 2
    metric[np.where(d_gt == -1)] = np.nan
    return metric


def rel_err(d, d_gt):
    metric = (d - d_gt) / (d_gt + np.finfo(float).eps)
    metric[np.where(d_gt == -1)] = np.nan
    return metric


def abs_rel_err(d, d_gt):
    metric = np.abs((d - d_gt) / (d_gt + np.finfo(float).eps))
    metric[np.where(d_gt == -1)] = np.nan
    return metric


def sq_rel_err(d, d_gt):
    metric = ((d - d_gt) ** 2) / (d_gt + np.finfo(float).eps)
    metric[np.where(d_gt == -1)] = np.nan
    return metric


def accuracy(d, d_gt, thresh):
    dev1 = np.abs(d / (d_gt + np.finfo(float).eps))
    dev2 = np.abs(d_gt / (d + np.finfo(float).eps))
    dev_max = np.maximum(dev1, dev2)
    dev_th = 1 * np.less(dev_max, thresh).astype('float32')
    dev_th[np.where(d_gt == -1)] = np.nan
    return dev_th


def class_accuracy(d, d_gt, seg, thresh, label_list):
    if seg is None:
        return None

    class_pix = np.in1d(seg, label_list).reshape(seg.shape)
    class_gt = np.copy(d_gt)
    class_gt[np.logical_not(class_pix)] = -1
    metric = accuracy(d, class_gt, thresh)
    return metric


def class_rel_err(d, d_gt, seg, label_list):
    if seg is None:
        return None

    class_pix = np.in1d(seg, label_list).reshape(seg.shape)
    class_gt = np.copy(d_gt)
    class_gt[np.logical_not(class_pix)] = -1
    metric = rel_err(d, class_gt)
    return metric


def calc_accuracy_from_relative_error(rel_err, delta_th):
    z_div_zgt = rel_err + 1.
    delta = np.maximum(np.abs(z_div_zgt), np.abs(1. / z_div_zgt))
    counts = (delta <= delta_th).astype(int).sum()
    accuracy = counts / len(rel_err)
    return counts, accuracy


def run_analysis(data_loader, analyzer, cfg, verbose=True):
    """Analyze required frames and dump results into a single pickle."""

    start_frm_idx = cfg.analysis.start_frm_idx
    end_frm_idx = cfg.analysis.end_frm_idx
    if end_frm_idx == -1:
        end_frm_idx = len(data_loader)
    assert end_frm_idx > start_frm_idx

    eval_mask_list = analyzer.eval_mask_list

    for i in range(start_frm_idx, end_frm_idx):
        if verbose:
            print("Processing frame: ", i)
        # img, d_gt, seg, d, pose, K, conf = data_loader[i]
        img, d_gt, d = data_loader[i]

        # metrics_results = analyzer.eval_frame(i, d_gt, seg, d, pose, K, conf)
        if not eval_mask_list is None:
            eval_mask = cv2.imread(eval_mask_list[i].split('\n')[0])[:,:,0]
        else:
            eval_mask = None

        metrics_results = analyzer.eval_frame(i, d_gt, d, eval_mask)

    analyzer.save_results(cfg.save.folder + '/sfm_analysis', cfg.analysis.out_file_name[0])



