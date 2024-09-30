# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC). All rights reserved.

import numpy as np

def get_ord(val1, val2, delta):
    ratio = (val1+0.00000000001)/(val2+0.00000000001)
    if ratio > delta:
        out = 1
    elif ratio < 1/delta:
        out = -1
    else:
        out = 0

    return out

def d3r(gt, depth_est, center_points, point_pairs):

    tol = 0.03

    center_y = center_points[0]
    center_x = center_points[1]

    center_y_selected_1 = center_y[point_pairs[:,0]]
    center_y_selected_2 = center_y[point_pairs[:,1]]
    center_x_selected_1 = center_x[point_pairs[:,0]]
    center_x_selected_2 = center_x[point_pairs[:,1]]

    # ord_arr = np.zeros((len(center_y_selected_1)))

    gt_selected_1 = gt[center_y_selected_1,center_x_selected_1]
    gt_selected_2 = gt[center_y_selected_2,center_x_selected_2]
    gt_ratio = gt_selected_1/gt_selected_2
    gt_sign_pos = gt_ratio > 1+tol
    gt_sign_neg = gt_ratio < 1-tol
    pred_selected_1 = depth_est[center_y_selected_1,center_x_selected_1]
    pred_selected_2 = depth_est[center_y_selected_2,center_x_selected_2]
    pred_ratio = pred_selected_1/pred_selected_2
    pred_sign_pos = pred_ratio > 1+tol
    pred_sign_neg = pred_ratio < 1-tol

    ord_ratio = gt_sign_pos*pred_sign_pos + gt_sign_neg*pred_sign_neg

    return ord_ratio
