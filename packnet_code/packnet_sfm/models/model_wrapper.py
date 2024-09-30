# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC), based on:
# https://github.com/TRI-ML/packnet-sfm - Toyota Research Institute

from collections import OrderedDict
import os
import time
import random
import numpy as np
import torch
import cv2
from torch.utils.data import ConcatDataset, DataLoader

from packnet_code.packnet_sfm.datasets.transforms import get_transforms
from packnet_code.packnet_sfm.utils.depth import inv2depth, post_process_inv_depth, compute_depth_metrics
from packnet_code.packnet_sfm.utils.horovod import print0, world_size, rank, on_rank_0
from packnet_code.packnet_sfm.utils.image import flip_lr
from packnet_code.packnet_sfm.utils.load import load_class, load_class_args_create, \
    load_network, filter_args
from packnet_code.packnet_sfm.utils.logging import pcolor
from packnet_code.packnet_sfm.utils.reduce import all_reduce_metrics, reduce_dict, \
    create_dict, average_loss_and_metrics
from packnet_code.packnet_sfm.utils.save import save_depth
from packnet_code.packnet_sfm.models.model_utils import stack_batch
from packnet_code.packnet_sfm.utils.edge import edge_from_depth, chamfer_distance

from packnet_code.packnet_sfm.losses.grad_loss import GradLoss



class ModelWrapper(torch.nn.Module):
    """
    Top-level torch.nn.Module wrapper around a SfmModel (pose+depth networks).
    Designed to use models with high-level Trainer classes (cf. trainers/).

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    """

    def __init__(self, config, resume=None, logger=None, load_datasets=True):
        super().__init__()

        # Store configuration, checkpoint and logger
        self.config = config
        self.logger = logger
        self.resume = resume

        # Set random seed
        set_random_seed(config.arch.seed)

        # Task metrics
        self.metrics_name = 'depth'
        self.metrics_keys = ('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
        self.metrics_modes = ('', '_pp', '_gt', '_pp_gt')

        self.metrics_edges_name = 'edges'
        self.metrics_edges_keys = ('precision_1', 'recall_1', 'precision_2', 'recall_2', 'precision_3', 'recall_3')
        self.metrics_edges_modes = ('', '_input')

        # Model, optimizers, schedulers and datasets are None for now
        self.model = self.optimizer = self.scheduler = None
        self.train_dataset = self.validation_dataset = self.test_dataset = None
        self.current_epoch = 0

        # Prepare model
        self.prepare_model(resume)

        # Prepare datasets
        if load_datasets:
            # Requirements for validation (we only evaluate depth for now)
            validation_requirements = {'gt_depth': True, 'gt_pose': False}
            test_requirements = validation_requirements
            self.prepare_datasets(validation_requirements, test_requirements)

        # Preparations done
        self.config.prepared = True

        # self.sigmoid = torch.nn.Sigmoid()

    def prepare_model(self, resume=None):
        """Prepare self.model (incl. loading previous state)"""
        print0(pcolor('### Preparing Model', 'green'))
        # self.model = setup_model(self.config.model, self.config.prepared)
        self.model = setup_model(self.config, self.config.prepared)
        # Resume model if available
        if resume:
            print0(pcolor('### Resuming from {}'.format(
                resume['file']), 'magenta', attrs=['bold']))
            self.model = load_network(
                self.model, resume['state_dict'], 'model')
            if 'epoch' in resume:
                self.current_epoch = resume['epoch'] + 1

    def prepare_datasets(self, validation_requirements, test_requirements):
        """Prepare datasets for training, validation and test."""
        # Prepare datasets
        print0(pcolor('### Preparing Datasets', 'green'))

        augmentation = self.config.datasets.augmentation
        # Setup train dataset (requirements are given by the model itself)
        self.train_dataset = setup_dataset(
            self.config.datasets.train, 'train',
            self.model.train_requirements, **augmentation)
        # Setup validation dataset
        self.validation_dataset = setup_dataset(
            self.config.datasets.validation, 'validation',
            validation_requirements, **augmentation)
        # Setup test dataset
        self.test_dataset = setup_dataset(
            self.config.datasets.test, 'test',
            test_requirements, **augmentation)

    @property
    def depth_net(self):
        """Returns depth network."""
        return self.model.depth_net

    @property
    def pose_net(self):
        """Returns pose network."""
        return self.model.pose_net

    @property
    def logs(self):
        """Returns various logs for tracking."""
        params = OrderedDict()
        for param in self.optimizer.param_groups:
            params['{}_learning_rate'.format(param['name'].lower())] = param['lr']
        params['progress'] = self.progress
        return {
            **params,
            **self.model.logs,
        }

    @property
    def progress(self):
        """Returns training progress (current epoch / max. number of epochs)"""
        return self.current_epoch / self.config.arch.max_epochs

    def configure_optimizers(self):
        """Configure depth and pose optimizers and the corresponding scheduler."""

        params = []
        # Load optimizer
        optimizer = getattr(torch.optim, self.config.model.optimizer.name)
        # Depth optimizer
        if self.depth_net is not None:
            params.append({
                'name': 'Depth',
                'params': self.depth_net.parameters(),
                **filter_args(optimizer, self.config.model.optimizer.depth)
            })
        # Pose optimizer
        if self.pose_net is not None:
            params.append({
                'name': 'Pose',
                'params': self.pose_net.parameters(),
                **filter_args(optimizer, self.config.model.optimizer.pose)
            })
        # Create optimizer with parameters
        optimizer = optimizer(params)

        # Load and initialize scheduler
        scheduler = getattr(torch.optim.lr_scheduler, self.config.model.scheduler.name)
        scheduler = scheduler(optimizer, **filter_args(scheduler, self.config.model.scheduler))

        if self.resume:
            if 'optimizer' in self.resume:
                optimizer.load_state_dict(self.resume['optimizer'])
            if 'scheduler' in self.resume:
                scheduler.load_state_dict(self.resume['scheduler'])

        # Create class variables so we can use it internally
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Return optimizer and scheduler
        return optimizer, scheduler

    def train_dataloader(self):
        """Prepare training dataloader."""
        return setup_dataloader(self.train_dataset,
                                self.config.datasets.train, 'train')[0]

    def val_dataloader(self):
        """Prepare validation dataloader."""
        return setup_dataloader(self.validation_dataset,
                                self.config.datasets.validation, 'validation')

    def test_dataloader(self):
        """Prepare test dataloader."""
        return setup_dataloader(self.test_dataset,
                                self.config.datasets.test, 'test')

    def training_step(self, batch, *args):
        """Processes a training batch."""
        batch = stack_batch(batch)
        output = self.model(batch, progress=self.progress)
        # contains: loss, inv_depths, inv_depths_rgbd, depth_loss(?), metrics-edge_loss, metrics-edge_lidar_loss
        # Log to wandb
        # i_num_of_batch_in_epoch, outputs_list_of_metrics_and_loss = args
        if self.logger:
            if len(args[1]) > 0 and args[0]%self.config.wandb.train_log_step == 0:
                loss_and_metrics = average_loss_and_metrics(args[1], 'avg_train')
                self.logger.log_metrics({
                    **self.logs, **loss_and_metrics,
                }, force_log=True)
        return {
            'loss': output['loss'],
            'metrics': output['metrics']
        }

    def validation_step(self, batch, *args):
        """Processes a validation batch."""
        output = self.evaluate_depth(batch, args)
        if self.logger:
            self.logger.log_depth('val', batch, output, args,
                                  self.validation_dataset, world_size(),
                                  self.config.datasets.validation)
        return {
            'idx': batch['idx'],
            **output['metrics'],
        }

    def test_step(self, batch, *args):
        """Processes a test batch."""
        output = self.evaluate_depth(batch, args)
        save_depth(batch, output, args,
                   self.config.datasets.test,
                   self.config.save)
        return {
            'idx': batch['idx'],
            **output['metrics'],
        }

    def training_epoch_end(self, output_batch):
        """Finishes a training epoch."""

        # Calculate and reduce average loss and metrics per GPU
        loss_and_metrics = average_loss_and_metrics(output_batch, 'avg_train')
        # loss_and_metrics = reduce_dict(loss_and_metrics, to_item=True)

        # Log to wandb
        if self.logger:
            self.logger.log_metrics({
                **self.logs, **loss_and_metrics,
            })

        return {
            **loss_and_metrics
        }

    def validation_epoch_end(self, output_data_batch):
        """Finishes a validation epoch."""

        # Reduce depth metrics
        metrics_data = all_reduce_metrics(
            output_data_batch, self.validation_dataset, self.metrics_name)

        # Create depth dictionary
        metrics_dict = create_dict(
            metrics_data, self.metrics_keys, self.metrics_modes,
            self.config.datasets.validation)

        # Print stuff
        self.print_metrics(metrics_data, self.config.datasets.validation)


        metrics_edges_data = all_reduce_metrics(
            output_data_batch, self.validation_dataset, self.metrics_edges_name)

        # # Create depth dictionary
        metrics_edges_dict = create_dict(
            metrics_edges_data, self.metrics_edges_keys, self.metrics_edges_modes,
            self.config.datasets.validation, name='edges')

        # Print stuff
        self.print_edges_metrics(metrics_edges_data, self.config.datasets.validation)


        # Log to wandb
        if self.logger:
            self.logger.log_metrics({
                **metrics_dict, 'global_step': self.current_epoch + 1,
            })

        return {
            **metrics_dict,
            **metrics_edges_dict
        }

    def test_epoch_end(self, output_data_batch):
        """Finishes a test epoch."""

        # Reduce depth metrics
        metrics_data = all_reduce_metrics(
            output_data_batch, self.test_dataset, self.metrics_name)

        # Create depth dictionary
        metrics_dict = create_dict(
            metrics_data, self.metrics_keys, self.metrics_modes,
            self.config.datasets.test)

        # Print stuff
        self.print_metrics(metrics_data, self.config.datasets.test)

        return {
            **metrics_dict
        }

    def forward(self, *args, **kwargs):
        """Runs the model and returns the output."""
        assert self.model is not None, 'Model not defined'
        return self.model(*args, **kwargs)

    def depth(self, *args, **kwargs):
        """Runs the pose network and returns the output."""
        assert self.depth_net is not None, 'Depth network not defined'
        return self.depth_net(*args, **kwargs)

    def pose(self, *args, **kwargs):
        """Runs the depth network and returns the output."""
        assert self.pose_net is not None, 'Pose network not defined'
        return self.pose_net(*args, **kwargs)

    def evaluate_depth(self, batch, args):
        """Evaluate batch to produce depth metrics."""
        # Get predicted depth
        inv_depths = self.model(batch)['inv_depths'][0]
        depth = inv2depth(inv_depths[0][:,0:1,:,:])
        # Post-process predicted depth
        batch['rgb'] = flip_lr(batch['rgb'])
        if 'input_depth' in batch:
            batch['input_depth'] = flip_lr(batch['input_depth'])
        if 'rgb_edge' in batch:
            batch['rgb_edge'] = flip_lr(batch['rgb_edge'])
        inv_depths_flipped = self.model(batch)['inv_depths'][0]
        inv_depth_pp = post_process_inv_depth(
            inv_depths[0][:,0:1,:,:], inv_depths_flipped[0][:,0:1,:,:], method='mean')
        depth_pp = inv2depth(inv_depth_pp)
        batch['rgb'] = flip_lr(batch['rgb'])
        # Calculate predicted metrics
        metrics = OrderedDict()
        if 'depth' in batch:
            for mode in self.metrics_modes:
                metrics[self.metrics_name + mode] = compute_depth_metrics(
                    self.config.model.params, gt=batch['depth'],
                    pred=depth_pp if 'pp' in mode else depth,
                    use_gt_scale='gt' in mode)
        if 'edge' in batch:
            if self.config.model.depth_net.input_channels==4:
                pred_depth_for_edges = self.depth(batch['rgb'],rgb_edge=batch['rgb_edge'])['inv_depths'][0][0]
            else:
                pred_depth_for_edges = self.depth(batch['rgb'])['inv_depths'][0][0]
            if 'input_depth' in batch:
                if self.config.model.depth_net.input_channels == 4:
                    pred_input_depth_for_edges = self.depth(batch['rgb'], batch['input_depth'], rgb_edge=batch['rgb_edge'])['inv_depths'][0][0]
                else:
                    pred_input_depth_for_edges = self.depth(batch['rgb'], batch['input_depth'])['inv_depths'][0][0]
            metrics['edges'] = []
            self.compute_edge_metrics(pred_depth_for_edges[:,0:1,:,:].detach(), batch['edge'].detach(), metrics['edges'], args[1])
            metrics['edges'] = torch.tensor(metrics['edges'])
            if 'input_depth' in batch:
                metrics['edges_input'] = []
                self.compute_edge_metrics(pred_input_depth_for_edges[:,0:1,:,:].detach(), batch['edge'].detach(), metrics['edges_input'], args[1])
                metrics['edges_input'] = torch.tensor(metrics['edges_input'])

        # Return metrics and extra information
        return {
            'metrics': metrics,
            'inv_depth': inv_depth_pp
        }

    def compute_edge_metrics(self, depth, edge, metrics_entry, dataset_idx):

        # pred_depth = inv2depth(depth)
        gt_edge = edge.detach().cpu().numpy()[0, 0, :, :] * 255
        new_shape = gt_edge.shape

        if not (self.config.model.name == 'EdgeEstimationLIDARModel' or self.config.model.name == 'EdgeEstimationModel' or self.config.model.name == 'EdgeClassificationLIDARModel'):
            pred_depth = inv2depth(depth)
            pred_depth = pred_depth.detach().cpu().numpy()[0, 0, :, :]

            if not new_shape is None:

                depth_resized = cv2.resize(pred_depth, (new_shape[1], new_shape[0]),
                                           interpolation=cv2.INTER_LINEAR)
            else:
                depth_resized = pred_depth
            # depth_mask = depth_resized < cfg.analysis.min_depth
            # depth_resized[depth_mask] = cfg.analysis.min_depth
            # depth_mask = depth_resized > cfg.analysis.max_depth
            # depth_resized[depth_mask] = cfg.analysis.max_depth
            depth_resized_vis = depth_resized * (255.0 / np.max(depth_resized))
            depth_resized_vis = depth_resized_vis.astype(np.uint8)
            # TODO: consider transfering to config
            edge_im_1 = cv2.Canny(depth_resized_vis, 10, 20)
            edge_im_2 = cv2.Canny(depth_resized_vis, 20, 40)
            edge_im_3 = cv2.Canny(depth_resized_vis, 30, 60)

        else:
            pred_depth_sigmoid = depth
            # pred_depth_sigmoid = self.sigmoid(pred_depth - 4)
            pred_depth_sigmoid = pred_depth_sigmoid.detach().cpu().numpy()[0, 0, :, :]
            if not new_shape is None:
                edge_pred_im = cv2.resize(pred_depth_sigmoid, (new_shape[1], new_shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
            else:
                edge_pred_im = pred_depth_sigmoid
            # TODO: consider transfering to config
            edge_im_1 = (edge_pred_im > 0.5).astype('uint8') * 255
            edge_im_2 = (edge_pred_im > 0.75).astype('uint8') * 255
            edge_im_3 = (edge_pred_im > 0.9).astype('uint8') * 255

        edge_images = [edge_im_1, edge_im_2, edge_im_3]

        # precision_arr = []
        # recall_arr = []

        if len(self.config.datasets.validation.gt_crop) > 0:
            # dataset_idx = args[0]
            crop = self.config.datasets.validation.gt_crop[dataset_idx]
            gt_edge = gt_edge[crop[2]:crop[3], crop[0]:crop[1]]

        for cur_edge_im in edge_images:
            if len(self.config.datasets.validation.gt_crop) > 0:
                # dataset_idx = args[0]
                crop = self.config.datasets.validation.gt_crop[dataset_idx]
                cur_edge_pred_im = cur_edge_im[crop[2]:crop[3], crop[0]:crop[1]]
            else:
                cur_edge_pred_im = cur_edge_im
            c_dist_1, c_per_1, _ = chamfer_distance(cur_edge_pred_im, gt_edge)
            metrics_entry.append(c_per_1)
            c_dist_2, c_per_2, _ = chamfer_distance(gt_edge, cur_edge_pred_im)
            metrics_entry.append(c_per_2)

            mean_f1_score = 2*((c_per_1*c_per_2)/(c_per_1+c_per_2))
            metrics_entry.append(mean_f1_score)

        return metrics_entry

    @on_rank_0
    def print_metrics(self, metrics_data, dataset):
        """Print depth metrics on rank 0 if available"""
        if not metrics_data[0]:
            return

        hor_line = '|{:<}|'.format('*' * 93)
        met_line = '| {:^14} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} |'
        num_line = '{:<14} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f}'

        def wrap(string):
            return '| {} |'.format(string)

        print()
        print()
        print()
        print(hor_line)

        if self.optimizer is not None:
            bs = 'E: {} BS: {}'.format(self.current_epoch + 1,
                                       self.config.datasets.train.batch_size)
            if self.model is not None:
                bs += ' - {}'.format(self.config.model.name)
            lr = 'LR ({}):'.format(self.config.model.optimizer.name)
            for param in self.optimizer.param_groups:
                lr += ' {} {:.2e}'.format(param['name'], param['lr'])
            par_line = wrap(pcolor('{:<40}{:>51}'.format(bs, lr),
                                   'green', attrs=['bold', 'dark']))
            print(par_line)
            print(hor_line)

        print(met_line.format(*(('METRIC',) + self.metrics_keys)))
        for n, metrics in enumerate(metrics_data):
            print(hor_line)
            path_line = '{}'.format(
                os.path.join(dataset.path[n], dataset.split[n]))
            if len(dataset.cameras[n]) == 1: # only allows single cameras
                path_line += ' ({})'.format(dataset.cameras[n][0])
            print(wrap(pcolor('*** {:<87}'.format(path_line), 'magenta', attrs=['bold'])))
            print(hor_line)
            for key, metric in metrics.items():
                if self.metrics_name in key:
                    print(wrap(pcolor(num_line.format(
                        *((key.upper(),) + tuple(metric.tolist()))), 'cyan')))
        print(hor_line)

        if self.logger:
            run_line = wrap(pcolor('{:<60}{:>31}'.format(
                self.config.wandb.url, self.config.wandb.name), 'yellow', attrs=['dark']))
            print(run_line)
            print(hor_line)

        print()

    def print_edges_metrics(self, metrics_data, dataset):
        """Print depth metrics on rank 0 if available"""
        if not metrics_data[0]:
            return

        hor_line = '|{:<}|'.format('*' * 93)
        met_line = '| {:^14} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | '
        num_line = '{:<14} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} '

        def wrap(string):
            return '| {} |'.format(string)

        print()
        print()
        print()
        print(hor_line)

        if self.optimizer is not None:
            bs = 'E: {} BS: {}'.format(self.current_epoch + 1,
                                       self.config.datasets.train.batch_size)
            if self.model is not None:
                bs += ' - {}'.format(self.config.model.name)
            lr = 'LR ({}):'.format(self.config.model.optimizer.name)
            for param in self.optimizer.param_groups:
                lr += ' {} {:.2e}'.format(param['name'], param['lr'])
            par_line = wrap(pcolor('{:<40}{:>51}'.format(bs, lr),
                                   'green', attrs=['bold', 'dark']))
            print(par_line)
            print(hor_line)

        print(met_line.format(*(('METRIC',) + self.metrics_edges_keys)))
        for n, metrics in enumerate(metrics_data):
            print(hor_line)
            path_line = '{}'.format(
                os.path.join(dataset.path[n], dataset.split[n]))
            if len(dataset.cameras[n]) == 1: # only allows single cameras
                path_line += ' ({})'.format(dataset.cameras[n][0])
            print(wrap(pcolor('*** {:<87}'.format(path_line), 'magenta', attrs=['bold'])))
            print(hor_line)
            for key, metric in metrics.items():
                if self.metrics_edges_name in key:
                    print(wrap(pcolor(num_line.format(
                        *((key.upper(),) + tuple(metric.tolist()))), 'cyan')))
        print(hor_line)

        if self.logger:
            run_line = wrap(pcolor('{:<60}{:>31}'.format(
                self.config.wandb.url, self.config.wandb.name), 'yellow', attrs=['dark']))
            print(run_line)
            print(hor_line)

        print()



def set_random_seed(seed):
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_depth_net(config, prepared, **kwargs):
    """
    Create a depth network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    depth_net : nn.Module
        Create depth network
    """
    print0(pcolor('DepthNet: %s' % config.name, 'yellow'))
    depth_net = load_class_args_create(config.name,
        paths=['packnet_code.packnet_sfm.networks.depth',],
        args={**config, **kwargs},
    )
    if not prepared and config.checkpoint_path is not '':
        depth_net = load_network(depth_net, config.checkpoint_path,
                                 ['depth_net', 'disp_network'])
    return depth_net

def setup_depth_edge_loss(config):
    grad_loss = GradLoss(config.edges.edge_loss_type,
                         config.edges.use_external_edges_for_loss,
                         config.edges.edge_loss_class_list_to_mask_out,
                         config.edges.depth_edges_loss_weight,
                         config.edges.depth_edge_loss_pos_to_neg_weight)

    return grad_loss


def setup_pose_net(config, prepared, **kwargs):
    """
    Create a pose network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    pose_net : nn.Module
        Created pose network
    """
    print0(pcolor('PoseNet: %s' % config.name, 'yellow'))
    pose_net = load_class_args_create(config.name,
        paths=['packnet_code.packnet_sfm.networks.pose',],
        args={**config, **kwargs},
    )
    if not prepared and config.checkpoint_path is not '':
        pose_net = load_network(pose_net, config.checkpoint_path,
                                ['pose_net', 'pose_network'])
    return pose_net

###### add edge net here


def setup_model(config, prepared, **kwargs):
    """
    Create a model

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    prepared : bool
        True if the model has been prepared before
    kwargs : dict
        Extra parameters for the model

    Returns
    -------
    model : nn.Module
        Created model
    """
    print0(pcolor('Model: %s' % config.model.name, 'yellow'))
    model = load_class(config.model.name, paths=['packnet_code.packnet_sfm.models',])(
        **{**config.model.loss, **kwargs})
    # Add depth network if required
    if 'depth_net' in model.network_requirements:
        model.add_depth_net(setup_depth_net(config.model.depth_net, prepared))
    # Add pose network if required
    if 'pose_net' in model.network_requirements:
        model.add_pose_net(setup_pose_net(config.model.pose_net, prepared))
    if config.edges.train_depth_edges:
    # if config.model.name == 'SemiSupEdgeCompletionModel' or config.model.name == 'SemiSupEdgeModel':
        model.add_edge_loss(setup_depth_edge_loss(config))
    # If a checkpoint is provided, load pretrained model
    if not prepared and config.model.checkpoint_path is not '':
        model = load_network(model, config.model.checkpoint_path, 'model')
        print('Pretrained model loaded')
    if config.is_multi_gpu:
        # for model in self.models:
        #     self.models[model] = torch.nn.DataParallel(self.models[model])
        if 'depth_net' in model.network_requirements:
            model.depth_net = torch.nn.DataParallel(model.depth_net)
        if 'pose_net' in model.network_requirements:
            model.pose_net = torch.nn.DataParallel(model.pose_net)
    # Return model
    return model


def setup_dataset(config, mode, requirements, **kwargs):
    """
    Create a dataset class

    Parameters
    ----------
    config : CfgNode
        Configuration (cf. configs/default_config.py)
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the dataset
    requirements : dict (string -> bool)
        Different requirements for dataset loading (gt_depth, gt_pose, etc)
    kwargs : dict
        Extra parameters for dataset creation

    Returns
    -------
    dataset : Dataset
        Dataset class for that mode
    """
    # If no dataset is given, return None
    if len(config.path) == 0:
        return None

    print0(pcolor('###### Setup %s datasets' % mode, 'red'))

    # Global shared dataset arguments
    dataset_args = {
        'back_context': config.back_context,
        'forward_context': config.forward_context,
        'data_transform': get_transforms(mode, **kwargs)
    }

    # Loop over all datasets
    datasets = []
    for i in range(len(config.split)):
        path_split = os.path.join(config.path[i], config.split[i])

        # Individual shared dataset arguments
        dataset_args_i = {
            'depth_type': config.depth_type[i] if 'gt_depth' in requirements else None,
            'input_depth_type': config.input_depth_type[i] if 'gt_depth' in requirements else None,
            'with_pose': 'gt_pose' in requirements,
        }

        # KITTI dataset
        if config.dataset[i] == 'KITTI':
            from packnet_code.packnet_sfm.datasets.kitti_dataset import KITTIDataset
            dataset = KITTIDataset(
                config.path[i], path_split,
                **dataset_args, **dataset_args_i,
            )
        elif config.dataset[i] == 'GTA':
            from packnet_code.packnet_sfm.datasets.gta_dataset import GTADataset
            dataset = GTADataset(
                config.path[i], path_split,
                **dataset_args, **dataset_args_i
            )
        elif not config.dataset[i] == 'DA':
            ValueError('Unknown dataset %d' % config.dataset[i])

        # Repeat if needed
        if 'repeat' in config and config.repeat[i] > 1:
            dataset = ConcatDataset([dataset for _ in range(config.repeat[i])])
        datasets.append(dataset)

        # Display dataset information
        bar = '######### {:>7}'.format(len(dataset))
        if 'repeat' in config:
            bar += ' (x{})'.format(config.repeat[i])
        bar += ': {:<}'.format(path_split)
        print0(pcolor(bar, 'yellow'))

    # If training, concatenate all datasets into a single one
    if mode == 'train' and not config.dataset[0] == 'DA':
        datasets = [ConcatDataset(datasets)]

    return datasets


def worker_init_fn(worker_id):
    """Function to initialize workers"""
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)


def get_datasampler(dataset, mode):
    """Distributed data sampler"""
    return torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=(mode=='train'),
        num_replicas=world_size(), rank=rank())


def setup_dataloader(datasets, config, mode):
    """
    Create a dataloader class

    Parameters
    ----------
    datasets : list of Dataset
        List of datasets from which to create dataloaders
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the dataloader

    Returns
    -------
    dataloaders : list of Dataloader
        List of created dataloaders for each input dataset
    """
    return [(DataLoader(dataset,
                        batch_size=config.batch_size, shuffle=False,
                        pin_memory=True,
                        num_workers=config.num_workers,
                        # num_workers=0,
                        worker_init_fn=worker_init_fn,
                        sampler=get_datasampler(dataset, mode))
             ) for dataset in datasets]
