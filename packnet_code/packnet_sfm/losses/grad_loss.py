# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC). All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
# import kornia.filters as kf
import numpy as np
from scipy import ndimage
import cv2

from packnet_code.packnet_sfm.losses.attention_loss import AttentionLossSingleMap, attention_loss2

class GradLayer(nn.Module):

    def __init__(self, is_sharpen=False):
        super(GradLayer, self).__init__()

        if not is_sharpen:
            kernel_v = [[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]
            kernel_h = [[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]
            kernel_lr = [[-2, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 2]]
            kernel_rl = [[0, 1, 2],
                        [-1, 0, 1],
                        [-2, -1, 0]]

        else:
            kernel_v = [[-1, -1, -1],
                        [-1, 8.05, -1],
                        [-1, -1, -1]]
            kernel_h = [[0, -1, 0],
                        [-1, 4.05, -1],
                        [0, -1, 0]]
            kernel_lr = [[-2, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 2]]
            kernel_rl = [[0, 1, 2],
                        [-1, 0, 1],
                        [-2, -1, 0]]

        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        kernel_lr = torch.FloatTensor(kernel_lr).unsqueeze(0).unsqueeze(0)
        kernel_rl = torch.FloatTensor(kernel_rl).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        self.weight_lr = nn.Parameter(data=kernel_lr, requires_grad=False).cuda()
        self.weight_rl = nn.Parameter(data=kernel_rl, requires_grad=False).cuda()

    def get_gray(self,x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x, normal=None):

        if x.shape[1] == 3:
            x = self.get_gray(x)

        if normal is None:
            x_v = F.conv2d(x, self.weight_v, padding=1)
            x_h = F.conv2d(x, self.weight_h, padding=1)
            x_mag = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        else:
            x_v = F.conv2d(x, self.weight_v, padding=1)
            x_h = F.conv2d(x, self.weight_h, padding=1)
            x_lr = F.conv2d(x, self.weight_lr, padding=1)
            x_rl = F.conv2d(x, self.weight_rl, padding=1)

            x_mag = torch.abs(x_h)
            range_v_neg = torch.logical_and(normal >= -5*np.pi/8, normal < -3*np.pi/8)
            range_v_pos = torch.logical_and(normal >= 3*np.pi/8, normal < 5*np.pi/8)
            range_v = torch.logical_or(range_v_neg, range_v_pos)
            x_mag[range_v] = torch.abs(x_v[range_v])

            range_rl_neg = torch.logical_and(normal >= -7*np.pi/8, normal < -5*np.pi/8)
            range_rl_pos = torch.logical_and(normal >= 1*np.pi/8, normal < 3*np.pi/8)
            range_rl = torch.logical_or(range_rl_neg, range_rl_pos)
            x_mag[range_rl] = torch.abs(x_rl[range_rl])
            range_lr_neg = torch.logical_and(normal >= -3*np.pi/8, normal < -1*np.pi/8)
            range_lr_pos = torch.logical_and(normal >= 5*np.pi/8, normal < 7*np.pi/8)
            range_lr = torch.logical_or(range_lr_neg, range_lr_pos)
            x_mag[range_lr] = torch.abs(x_lr[range_lr])

        return x_mag, x_v, x_h

class GradLoss(nn.Module):

    def __init__(self,
                 edge_loss_type,
                 use_external_edges_for_loss=True,
                 edge_loss_class_list_to_mask_out=[],
                 depth_edges_loss_weight=1.0,
                 depth_edges_loss_pos_to_neg_weight=1.0):

        super(GradLoss, self).__init__()
        self.loss_l1 = nn.L1Loss(reduce=False)
        self.grad_layer = GradLayer()
        self.sharpen_layer = GradLayer(is_sharpen=True)

        self.weight = depth_edges_loss_weight
        self.depth_edges_loss_pos_to_neg_weight = depth_edges_loss_pos_to_neg_weight
        self.bce_loss = nn.BCELoss(reduce=False)

        self.device = 'cuda'
        self.edge_loss_type = edge_loss_type
        self.use_external_edges_for_loss = use_external_edges_for_loss
        self.edge_loss_class_list_to_mask_out = edge_loss_class_list_to_mask_out

        self.erode_kernel = np.ones((3, 3), np.uint8)

    def forward(self, output, gt_edge, gt_mask=None, is_grad=True, is_sigmoid=True, sigmoid_thresh=4, gt_normals=None):


        b_size, chan_size, gt_h, gt_w = gt_edge.shape

        output = F.interpolate(output, size=(gt_h, gt_w), mode='bilinear')

        if is_grad:
            output_grad, _, _ = self.grad_layer(output, gt_normals)
        else:
            output_grad = output

        if is_sigmoid:
            output_edge_prob = torch.sigmoid(output_grad - sigmoid_thresh)
        else:
            output_edge_prob = output_grad

        # log for positives, square loss for negatives
        if 'cross_entropy' in self.edge_loss_type:
            edge_loss = self.comp_cross_entropy(gt_edge, gt_mask, output_edge_prob)

        if 'attention_loss' in self.edge_loss_type:
            edge_loss = attention_loss2(output_edge_prob, gt_edge, gt_mask, False)

        if 'spatially_adaptive' in self.edge_loss_type:
            edge_loss = attention_loss2(output_edge_prob, gt_edge, gt_mask, True)

        # dice loss
        if 'dice' in self.edge_loss_type:

            num_loss_elem = torch.numel(gt_edge)
            # 1000 weight is based on https://openaccess.thecvf.com/content_ECCV_2018/papers/Ruoxi_Deng_Learning_to_Predict_ECCV_2018_paper.pdf
            dice_loss = 1000*((torch.sum(output_edge_prob**2) + torch.sum(gt_edge**2)+0.0001)/(2*(torch.sum(output_edge_prob*gt_edge))+0.0001))/num_loss_elem

            edge_loss += dice_loss

        edge_loss = self.weight*(edge_loss.mean())
        return edge_loss, output_grad.detach()

    def comp_cross_entropy(self, gt_edge, gt_mask, output_edge_prob):


        if gt_mask is None:
            gt_mask = torch.ones_like(gt_edge)

        output_edge_pos_loss = -gt_edge * torch.log(output_edge_prob + 0.001)

        weights_pos = torch.sum(gt_edge * gt_mask, dim=(1, 2, 3))

        neg_mask = torch.ones_like(gt_edge) - gt_edge

        weights_neg = torch.sum(neg_mask * gt_mask, dim=(1, 2, 3))

        if weights_neg.sum()==0:
            alpha_vec = torch.ones_like(weights_neg)
        else:
            alpha_vec = weights_neg/(weights_pos+weights_neg)

        output_edge_neg_loss = -neg_mask * torch.log(1 - output_edge_prob + 0.001)

        edge_loss_class_list_to_mask_out = []
        if (torch.unique(gt_mask).shape[0] == 2) & bool(torch.any(torch.unique(gt_mask) == 1)) & bool(
                torch.any(torch.unique(gt_mask) == 0)):
            output_edge_pos_loss[gt_mask == 0] = 0
            output_edge_neg_loss[gt_mask == 0] = 0
            valid_pixels_num = gt_mask.sum()
        elif len(edge_loss_class_list_to_mask_out) > 0 and (not gt_mask is None):

            gt_mask_uint8 = (gt_mask * 255).type(torch.IntTensor).cuda()

            gt_mask_bin = torch.ones_like(gt_edge)
            for cur_mask_idx in range(0, len(edge_loss_class_list_to_mask_out)):
                cur_seg = edge_loss_class_list_to_mask_out[cur_mask_idx]
                cur_gt_mask_bin = torch.unsqueeze(torch.logical_not(torch.logical_and(
                    torch.logical_and(gt_mask_uint8[:, 0, :, :] == cur_seg[0],
                                      gt_mask_uint8[:, 1, :, :] == cur_seg[1]),
                    gt_mask_uint8[:, 2, :, :] == cur_seg[2])), dim=1)
                cur_gt_mask_bin_numpy = cur_gt_mask_bin.detach().cpu().numpy()[0, 0, :, :].astype('uint8')
                # we want to erode, but this is all 1s and the 0s are where the trees are
                cur_gt_mask_bin_numpy = cv2.dilate(cur_gt_mask_bin_numpy, self.erode_kernel)
                cur_gt_mask_bin = torch.from_numpy(cur_gt_mask_bin_numpy).type(torch.FloatTensor).cuda()
                gt_mask_bin = gt_mask_bin * cur_gt_mask_bin

            output_edge_pos_loss = output_edge_pos_loss * gt_mask_bin
            output_edge_neg_loss = output_edge_neg_loss * gt_mask_bin
            valid_pixels_num = gt_mask_bin.sum()
        else:
            valid_pixels_num = torch.numel(
                gt_edge)

        edge_loss_mat = (self.depth_edges_loss_pos_to_neg_weight * alpha_vec * torch.sum(output_edge_pos_loss,
                                                                                         dim=(1, 2, 3)) +
                         (1 - alpha_vec) * torch.sum(output_edge_neg_loss, dim=(1, 2, 3))).sum()

        edge_loss = edge_loss_mat / valid_pixels_num


        return edge_loss
