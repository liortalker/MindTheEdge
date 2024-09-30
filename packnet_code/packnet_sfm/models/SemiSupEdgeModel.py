# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC), based on:
# https://github.com/TRI-ML/packnet-sfm - Toyota Research Institute

import torch

from packnet_code.packnet_sfm.models.SelfSupModel import SfmModel, SelfSupModel
from packnet_code.packnet_sfm.losses.supervised_loss import SupervisedLoss
# from packnet_code.packnet_sfm.losses.grad_loss import GradLoss
from packnet_code.packnet_sfm.models.model_utils import merge_outputs
from packnet_code.packnet_sfm.utils.depth import depth2inv, inv2depth


class SemiSupEdgeModel(SelfSupModel):
    """
    Model that inherits a depth and pose networks, plus the self-supervised loss from
    SelfSupModel and includes a supervised loss for semi-supervision.

    Parameters
    ----------
    supervised_loss_weight : float
        Weight for the supervised loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_loss_weight=0.9, depth_edges_loss_weight=10.0, **kwargs):
        # Initializes SelfSupModel
        super().__init__(**kwargs)
        # If supervision weight is 0.0, use SelfSupModel directly
        assert 0. < supervised_loss_weight <= 1., "Model requires (0, 1] supervision"
        # Store weight and initializes supervised loss
        self.supervised_loss_weight = supervised_loss_weight
        # self.depth_edges_loss_weight = depth_edges_loss_weight
        self._supervised_loss = SupervisedLoss(**kwargs)

        # Pose network is only required if there is self-supervision
        if self.supervised_loss_weight == 1:
            self._network_requirements.remove('pose_net')
        # GT depth is only required if there is supervision
        if self.supervised_loss_weight > 0:
            self._train_requirements.append('gt_depth')

        self.edges_depth_edge_loss_all_scales = kwargs['edges_depth_edge_loss_all_scales']
        self._input_keys = ['rgb', 'input_depth', 'edge', 'rgb_edge', 'normal']
        if self.edges_depth_edge_loss_all_scales:
            self._input_keys.append('edge_1')
            self._input_keys.append('edge_2')
            self._input_keys.append('edge_3')
            self._input_keys.append('normal_1')
            self._input_keys.append('normal_2')
            self._input_keys.append('normal_3')

        # self.weight_rgbd = weight_rgbd

        self.depth_edges_loss_weight = depth_edges_loss_weight

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._supervised_loss.logs
        }

    def supervised_loss(self, inv_depths, gt_inv_depths,
                        return_logs=False, progress=0.0):
        """
        Calculates the supervised loss.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        gt_inv_depths : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth maps from the original image
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._supervised_loss(
            inv_depths, gt_inv_depths,
            return_logs=return_logs, progress=progress)

    # def edge_loss(self, pred_depth, gt_depth, gt_mask, gt_edges):
    #
    #     return self.edge_loss_head(pred_depth, gt_depth, gt_mask, gt_edges)

    def edge_loss(self, pred_depth, gt_edges, gt_mask=None, is_grad=True, is_sigmoid=True, sigmoid_thresh=4, gt_normals=None):

        return self.edge_loss_head(pred_depth, gt_edges, gt_mask, is_grad, is_sigmoid, sigmoid_thresh, gt_normals)

    def forward(self, batch, return_logs=False, progress=0.0, **kwargs):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        if not self.training:
            # If not training, no need for self-supervised loss
            return SfmModel.forward(self, batch, return_logs=return_logs, **kwargs)
        else:
            if self.supervised_loss_weight == 1.:
                # If no self-supervision, no need to calculate loss
                self_sup_output = SfmModel.forward(self, batch, return_logs=return_logs, **kwargs)
                loss = torch.tensor([0.]).type_as(batch['rgb'])
            else:
                # Otherwise, calculate and weight self-supervised loss
                self_sup_output = SelfSupModel.forward(
                    self, batch, return_logs=return_logs, progress=progress, **kwargs)
                loss = (1.0 - self.supervised_loss_weight) * self_sup_output['loss']
            predicted_depth = self_sup_output['inv_depths']

            if 'rgb_edge' in batch.keys():
                rgb_edge_mask = batch['rgb_edge']
            else:
                rgb_edge_mask = None

            edge_loss = self.compute_edge_loss_with_all_scales(predicted_depth,
                                                               batch,
                                                               rgb_edge_mask,
                                                               is_grad=True,
                                                               is_sigmoid=True,
                                                               sigmoid_thresh=4)
            # Calculate and weight supervised loss
            sup_output = self.supervised_loss(
                self_sup_output['inv_depths'], depth2inv(batch['depth']),
                return_logs=return_logs, progress=progress)
            supervised_loss = self.supervised_loss_weight * sup_output['loss']
            loss += supervised_loss

            edge_loss = self.depth_edges_loss_weight*edge_loss
            loss += edge_loss
            # edge_dict = {}
            edge_output = {}
            edge_output['metrics'] = {}
            edge_output['metrics']['edge_loss'] = edge_loss.detach()
            edge_output['metrics']['supervised_loss'] = supervised_loss.detach()
            # sup_output['metrics']['edge_loss'] = edge_loss.detach()
            # Merge and return outputs
            return {
                'loss': loss,
                **merge_outputs(self_sup_output, edge_output),
            }

    def compute_edge_loss_with_all_scales(self, depths_data, batch, seg_mask,
                                          is_grad=False, is_sigmoid=False, sigmoid_thresh=4):
        predicted_rgb_depth = inv2depth(depths_data[0])
        if 'normal' in batch:
            normals = batch['normal']
        else:
            normals = None
        edge_rgb_loss, output_grad = self.edge_loss(predicted_rgb_depth,
                                                    batch['edge'],
                                                    gt_mask=seg_mask,
                                                    is_grad=is_grad,
                                                    is_sigmoid=is_sigmoid,
                                                    sigmoid_thresh=sigmoid_thresh,
                                                    gt_normals=normals)

        if self.edges_depth_edge_loss_all_scales:
            for cur_scale in range(1, 4):
                normal_key = 'normal_' + str(cur_scale)
                if normal_key in batch:
                    normals = batch[normal_key]
                else:
                    normals = None
                # cur_predicted_rgb_depth = inv2depth(self_sup_output['inv_depths'][cur_scale])
                cur_predicted_rgb_depth = inv2depth(depths_data[cur_scale])
                cur_edge_rgb_loss, cur_output_grad = self.edge_loss(cur_predicted_rgb_depth,
                                                                    batch['edge_' + str(cur_scale)],
                                                                    gt_mask=seg_mask,
                                                                    is_grad=is_grad,
                                                                    is_sigmoid=is_sigmoid,
                                                                    sigmoid_thresh=sigmoid_thresh,
                                                                    gt_normals=normals)
                edge_rgb_loss += cur_edge_rgb_loss

            edge_rgb_loss = edge_rgb_loss / 4
        return edge_rgb_loss