# MIT License
# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from Pytorch-Lightning
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/callbacks/model_checkpoint.py

import os, re
import numpy as np
import torch
from packnet_code.packnet_sfm.utils.logging import pcolor
import shutil

def sync_s3_data(local, model):
    """Sync saved models with the s3 bucket"""
    remote = os.path.join(model.config.checkpoint.s3_path, model.config.name)
    command = 'aws s3 sync {} {} --acl bucket-owner-full-control --quiet --delete'.format(local, remote)
    os.system(command)


def save_code(filepath):
    """Save code in the models folder"""
    os.system('tar cfz {}/code.tar.gz *'.format(filepath))


class ModelCheckpoint:
    def __init__(self, filepath=None, monitor='val_loss',
                 save_top_k=1, mode='auto', period=1,
                 s3_path='', s3_frequency=5, yaml_path=None):
        super().__init__()
        # If save_top_k is zero, save all models
        if save_top_k == 0:
            save_top_k = 1e6
        # Create checkpoint folder
        self.dirpath, self.filename = os.path.split(filepath)
        os.makedirs(self.dirpath, exist_ok=True)
        # Store arguments
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.period = period
        self.epoch_last_check = None
        self.best_k_models = {}
        self.kth_best_model = ''
        self.best = 0
        # Monitoring modes
        torch_inf = torch.tensor(np.Inf)
        mode_dict = {
            'min': (torch_inf, 'min'),
            'max': (-torch_inf, 'max'),
            'auto': (-torch_inf, 'max') if \
                'acc' in self.monitor or \
                'a1' in self.monitor or \
                self.monitor.startswith('fmeasure')
            else (torch_inf, 'min'),
        }
        self.kth_value, self.mode = mode_dict[mode]

        self.s3_path = s3_path
        self.s3_frequency = s3_frequency
        self.s3_enabled = s3_path is not '' and s3_frequency > 0
        self.save_code = True

        if yaml_path is not None and yaml_path!='' :
            shutil.copy2(yaml_path, self.dirpath)
            pass

    @staticmethod
    def _del_model(filepath):
        if os.path.isfile(filepath):
            os.remove(filepath)

    def _save_model(self, filepath, model):
        # Create folder, save model and sync to s3
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'config': model.config,
            'epoch': model.current_epoch,
            'state_dict': model.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'scheduler': model.scheduler.state_dict(),
        }, filepath)
        self._sync_s3(filepath, model)

    def _sync_s3(self, filepath, model):
        # If it's not time to sync, do nothing
        if self.s3_enabled and (model.current_epoch + 1) % self.s3_frequency == 0:
            filepath = os.path.dirname(filepath)
            # Print message and links
            print(pcolor('###### Syncing: {} -> {}'.format(filepath,
                model.config.checkpoint.s3_path), 'red', attrs=['bold']))
            print(pcolor('###### URL: {}'.format(
                model.config.checkpoint.s3_url), 'red', attrs=['bold']))
            # If it's time to save code
            if self.save_code:
                self.save_code = False
                save_code(filepath)
            # Sync model to s3
            sync_s3_data(filepath, model)

    def check_monitor_top_k(self, current):
        # If we don't have enough models
        if len(self.best_k_models) < self.save_top_k:
            return True
        # Convert to torch if necessary
        if not isinstance(current, torch.Tensor):
            current = torch.tensor(current)
        # Get monitoring operation
        monitor_op = {
            "min": torch.lt,
            "max": torch.gt,
        }[self.mode]
        # Compare and return
        return monitor_op(current, self.best_k_models[self.kth_best_model])

    def format_checkpoint_name(self, epoch, metrics):
        metrics['epoch'] = epoch
        filename = self.filename
        for tmp in re.findall(r'(\{.*?)[:\}]', self.filename):
            name = tmp[1:]
            filename = filename.replace(tmp, name + '={' + name)
            if name not in metrics:
                metrics[name] = 0
        filename = filename.format(**metrics)
        return os.path.join(self.dirpath, '{}.ckpt'.format(filename))

    def edges_format_checkpoint_name(self, epoch, metrics):

        filepath = self.format_checkpoint_name(epoch, metrics)

        metric_key_list = np.array(list(metrics.keys()))
        metric_val_list = np.array(list(metrics.values()))

        precision_rgb_indexes = np.where([('precision' in x) for x in metric_key_list])[0]
        precision_rgb_values = metric_val_list[precision_rgb_indexes]
        precision_lidar_indexes = np.where([('precision' in x) and ('input' in x) for x in metric_key_list])[0]
        precision_lidar_values = metric_val_list[precision_lidar_indexes]

        recall_rgb_indexes = np.where([('recall' in x) for x in metric_key_list])[0]
        recall_rgb_values = metric_val_list[recall_rgb_indexes]
        recall_lidar_indexes = np.where([('recall' in x) and ('input' in x) for x in metric_key_list])[0]
        recall_lidar_values = metric_val_list[recall_lidar_indexes]

        f1_rgb = np.mean(2 * ((precision_rgb_values * recall_rgb_values) / (precision_rgb_values + recall_rgb_values)))
        f1_lidar = np.mean(
            2 * ((precision_lidar_values * recall_lidar_values) / (precision_lidar_values + recall_lidar_values)))

        filepath = filepath.split('.ckpt')[0] + \
                   '_rgb_f1_' + str(np.round(f1_rgb, 4)) + '_lidar_f1_' + str(np.round(f1_lidar, 4)) + '.ckpt'

        return filepath


    def check_and_save(self, model, metrics):
        # Check saving interval
        epoch = model.current_epoch
        if self.epoch_last_check is not None and \
                (epoch - self.epoch_last_check) < self.period:
            return
        self.epoch_last_check = epoch
        # Prepare filepath

        # for metric in metrics:
        if 'EdgeEstimation' in model.config['model']['name'] or 'EdgeClassification' in model.config['model']['name']:
            filepath = self.edges_format_checkpoint_name(epoch, metrics)
        else:
            # filepath = self.format_checkpoint_name(epoch, metrics)
            filepath = self.edges_format_checkpoint_name(epoch, metrics)
        # else:
        #     precision_rgb_indexes = np.where([('abs_rel' in x) for x in metric_key_list])[0]
        #     filepath = self.dirpath + '/' + self.filename.split('.')[0] +\
        #                '_rgb_' + str(np.round(f1_rgb,2)) + '_lidar_' + str(np.round(f1_lidar,2))


        while os.path.isfile(filepath):
            #self.format_checkpoint_name(epoch, metrics)
            filepath = filepath[:-5] + 'b' + filepath[-5:] # add b at the end of the filename (instead of infinite loop)
        # Check if saving or not
        if self.save_top_k != -1:
            current = metrics.get(self.monitor)
            assert current, 'Checkpoint metric is not available'
            if self.check_monitor_top_k(current):
                self._do_check_save(filepath, model, current)
        else:
            self._save_model(filepath, model)
        for i in range(epoch):
            if 'EdgeEstimation' in model.config['model']['name'] or 'EdgeClassification' in model.config['model']['name']:
                delfilepath = self.edges_format_checkpoint_name(i, metrics)
            else:
                delfilepath = self.format_checkpoint_name(i, metrics)
            # if os.path.exists(delfilepath) and (i%model.config.checkpoint.save_freq != 0 or i==0):
            if os.path.exists(delfilepath) and (i % model.config.checkpoint.save_freq != 0):
                os.remove(delfilepath)

    def _do_check_save(self, filepath, model, current):
        # List of models to delete
        del_list = []
        if len(self.best_k_models) == self.save_top_k and self.save_top_k > 0:
            delpath = self.kth_best_model
            self.best_k_models.pop(self.kth_best_model)
            del_list.append(delpath)
        # Monitor current models
        self.best_k_models[filepath] = current
        if len(self.best_k_models) == self.save_top_k:
            # Monitor dict has reached k elements
            _op = max if self.mode == 'min' else min
            self.kth_best_model = _op(self.best_k_models,
                                      key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model]
        # Determine best model
        _op = min if self.mode == 'min' else max
        self.best = _op(self.best_k_models.values())
        # Delete old models
        for cur_path in del_list:
            if cur_path != filepath:
                self._del_model(cur_path)
        # Save model
        self._save_model(filepath, model)
