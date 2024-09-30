# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC), based on:
# https://github.com/TRI-ML/packnet-sfm - Toyota Research Institute

import torch

from packnet_code.packnet_sfm.models.SelfSupModel import SfmModel, SelfSupModel
from packnet_code.packnet_sfm.losses.supervised_loss import SupervisedLoss
from packnet_code.packnet_sfm.models.model_utils import merge_outputs
from packnet_code.packnet_sfm.utils.depth import depth2inv, inv2depth


class EdgeEstimationLIDARModel(SfmModel):
    """
    Semi-Supervised model for depth prediction and completion.

    Parameters
    ----------
    supervised_loss_weight : float
        Weight for the supervised loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_loss_weight=0.0, weight_rgbd=1.0, **kwargs):
        # Initializes SelfSupModel
        super().__init__(**kwargs)
        # Store weight and initializes supervised loss
        self.supervised_loss_weight = supervised_loss_weight
        # unused self._supervised_loss = SupervisedLoss(**kwargs)
        self.sigmoid = torch.nn.Sigmoid()
        # Pose network is only required if there is self-supervision
        self._network_requirements.remove('pose_net')
        self._train_requirements.append('gt_depth')
        # GT depth is only required if there is supervision
        # if self.supervised_loss_weight > 0:
        #     self._train_requirements.append('gt_depth')

        self.edges_depth_edge_loss_all_scales = kwargs['edges_depth_edge_loss_all_scales']
        self._input_keys = ['rgb', 'input_depth', 'edge']
        if self.edges_depth_edge_loss_all_scales:
            self._input_keys.append('edge_1')
            self._input_keys.append('edge_2')
            self._input_keys.append('edge_3')

        self.weight_rgbd = weight_rgbd



    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs
            # **self._supervised_loss.logs
        }

    # def supervised_loss(self, inv_depths, gt_inv_depths,
    #                     return_logs=False, progress=0.0):
    #     """
    #     Calculates the supervised loss.
    #
    #     Parameters
    #     ----------
    #     inv_depths : torch.Tensor [B,1,H,W]
    #         Predicted inverse depth maps from the original image
    #     gt_inv_depths : torch.Tensor [B,1,H,W]
    #         Ground-truth inverse depth maps from the original image
    #     return_logs : bool
    #         True if logs are stored
    #     progress :
    #         Training progress percentage
    #
    #     Returns
    #     -------
    #     output : dict
    #         Dictionary containing a "loss" scalar a "metrics" dictionary
    #     """
    #     return self._supervised_loss(
    #         inv_depths, gt_inv_depths,
    #         return_logs=return_logs, progress=progress)


    def edge_loss(self, pred_depth, gt_edges, gt_mask=None, is_grad=True, is_sigmoid=False, sigmoid_thresh=0):

        return self.edge_loss_head(pred_depth, gt_edges, gt_mask, is_grad, is_sigmoid, sigmoid_thresh)

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

        # normalize lidar
        if 'input_depth' in batch.keys():
            # LT: Why 200? that's a good question...
            batch['input_depth'] = batch['input_depth']/200.0

        self_sup_output = SfmModel.forward(self, batch, return_logs=return_logs, **kwargs)
        # contains: inv_depths, inv_depths_rgbd, depth_loss(?)
        if not self.training:
            num_scales = 1
        elif self.edges_depth_edge_loss_all_scales:
            num_scales = 4
        else:
            num_scales = 1

        for cur_scale in range(0, num_scales):
            if isinstance(self_sup_output['inv_depths'][cur_scale], list):
                self_sup_output['inv_depths'][0][cur_scale] = self_sup_output['inv_depths'][0][cur_scale] / 2
            else:
                self_sup_output['inv_depths'][cur_scale] = self_sup_output['inv_depths'][cur_scale] / 2
            # if 'input_depth' in batch.keys():
            if 'inv_depths_rgbd' in self_sup_output.keys():
                if isinstance(self_sup_output['inv_depths_rgbd'][cur_scale], list):
                    self_sup_output['inv_depths_rgbd'][0][cur_scale] = self_sup_output['inv_depths_rgbd'][0][cur_scale] / 2
                else:
                    self_sup_output['inv_depths_rgbd'][cur_scale] = self_sup_output['inv_depths_rgbd'][cur_scale] / 2

        if not self.training:
            return self_sup_output

        loss = torch.tensor([0.]).type_as(batch['rgb'])


        edge_rgb_loss = self.compute_edge_loss_with_all_scales(self_sup_output['inv_depths'],
                                                               batch, None, is_grad=False, is_sigmoid=False)

        if 'inv_depths_rgbd' in self_sup_output:
            edge_lidar_loss = self.compute_edge_loss_with_all_scales(self_sup_output['inv_depths_rgbd'],
                                                               batch, None, is_grad=False, is_sigmoid=False)

            if 'depth_loss' in self_sup_output:
                loss += self_sup_output['depth_loss']

        else:
            edge_lidar_loss = 0.0

        loss += (edge_rgb_loss + self.weight_rgbd * edge_lidar_loss)/2

        edge_output = {}
        edge_output['metrics'] = {}
        edge_output['metrics']['edge_loss'] = edge_rgb_loss.detach()
        if 'inv_depths_rgbd' in self_sup_output:
            edge_output['metrics']['edge_lidar_loss'] = edge_lidar_loss.detach()

        # Merge and return outputs
        return {
            'loss': loss,
            **merge_outputs(self_sup_output, edge_output),
        }

    def compute_edge_loss_with_all_scales(self, depths_data, batch, seg_mask, is_grad=False, is_sigmoid=False):
        predicted_rgb_depth = depths_data[0]
        edge_rgb_loss, output_grad = self.edge_loss(predicted_rgb_depth, batch['edge'], gt_mask=seg_mask,
                                                    is_grad=is_grad, is_sigmoid=is_sigmoid)

        if self.edges_depth_edge_loss_all_scales:
            for cur_scale in range(1, 4):
                cur_predicted_rgb_depth = depths_data[cur_scale]
                cur_edge_rgb_loss, cur_output_grad = self.edge_loss(cur_predicted_rgb_depth,
                                                                    batch['edge_' + str(cur_scale)], gt_mask=seg_mask,
                                                                    is_grad=is_grad,
                                                                    is_sigmoid=is_sigmoid)
                edge_rgb_loss += cur_edge_rgb_loss

            edge_rgb_loss = edge_rgb_loss / 4
        return edge_rgb_loss
