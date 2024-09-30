# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC), based on:
# https://github.com/TRI-ML/packnet-sfm - Toyota Research Institute

import os
import torch
import numpy as np
# import horovod.torch as hvd
# from packnet_code.packnet_sfm import BaseTrainer, sample_to_cuda
# from packnet_code.packnet_sfm import prep_logger_and_checkpoint
# from packnet_code.packnet_sfm import print_config
# from packnet_code.packnet_sfm import AvgMeter

# import packnet_code.packnet_sfm.utils.horovod.torch as hvd
from packnet_code.packnet_sfm.trainers.base_trainer import BaseTrainer, sample_to_cuda
from packnet_code.packnet_sfm.utils.config import prep_logger_and_checkpoint
from packnet_code.packnet_sfm.utils.logging import print_config
from packnet_code.packnet_sfm.utils.logging import AvgMeter

#import matplotlib.pyplot as plt

class CommonTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # hvd.init()
        # torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
        # torch.cuda.set_device(torch.device("cuda"))
        torch.backends.cudnn.benchmark = True

        self.avg_loss = AvgMeter(50)
        self.dtype = kwargs.get("dtype", None)  # just for test for now

    @property
    def proc_rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def fit(self, module):

        # Prepare module for training
        module.trainer = self
        # Update and print module configuration
        prep_logger_and_checkpoint(module)
        #print('Before print config')
        print_config(module.config)
        #print('After print config')

        # Send module to GPU
        module = module.to('cuda')
        # Configure optimizer and scheduler
        module.configure_optimizers()

        # Create distributed optimizer
        # compression = hvd.Compression.none
        # optimizer = hvd.DistributedOptimizer(module.optimizer,
        #     named_parameters=module.named_parameters(), compression=compression)
        # compression = hvd.Compression.none
        optimizer = module.optimizer
        scheduler = module.scheduler

        #print('Before data loadedrs')
        # Get train and val dataloaders
        train_dataloader = module.train_dataloader()
        val_dataloaders = module.val_dataloader()
        #print('After data loadedrs')

        # Validate before training if requested
        if self.validate_first:
            validation_output = self.validate(val_dataloaders, module)
            self.check_and_save(module, validation_output)

        all_validation_output = []
        # Epoch loop
        for epoch in range(module.current_epoch, self.max_epochs):
            # Train
            self.train(train_dataloader, module, optimizer)
            # Validation
            validation_output = self.validate(val_dataloaders, module)
            all_validation_output.append(list(validation_output.values()))
            # self.save_validation_graph(np.array(list(validation_output.keys())), np.array(all_validation_output))
            if epoch%1 == 0: # not module.config.checkpoint.save_freq, save every 1 to resume
                # Check and save model
                self.check_and_save(module, validation_output)
            # Update current epoch
            module.current_epoch += 1
            # Take a scheduler step
            scheduler.step()

    def train(self, dataloader, module, optimizer):
        # Set module to train
        module.train()
        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)
        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        # Start training loop
        outputs = []
        # For all batches
        all_loss_vec = []
        supervised_loss_vec = []
        edge_loss_vec = []
        edge_lidar_loss_vec = []
        normal_loss_vec = []
        normal_lidar_loss_vec = []
        for i, batch in progress_bar:
            # if i < 2600:
            #     continue
            # if i > 10:
            #     break
            # if i < 6688:
            #     continue
            # Reset optimizer
            optimizer.zero_grad()
            # Send samples to GPU and take a training stepp
            batch = sample_to_cuda(batch)
            output = module.training_step(batch, i, outputs)
            # Backprop through loss and take an optimizer step
            output['loss'].backward()
            optimizer.step()
            # Append output to list of outputs
            output['loss'] = output['loss'].detach()
            all_loss_vec.append(output['loss'].item())
            if 'supervised_loss' in output['metrics']:
                supervised_loss_vec.append(output['metrics']['supervised_loss'].item())
            else:
                supervised_loss_vec.append(-1)
            if 'edge_loss' in output['metrics']:
                cur_edge_loss_val = output['metrics']['edge_loss'].item()
            else:
                cur_edge_loss_val = -1
            if 'edge_lidar_loss' in output['metrics']:
                cur_edge_lidar_loss_val = output['metrics']['edge_lidar_loss'].item()
            else:
                cur_edge_lidar_loss_val = -1
            edge_loss_vec.append(cur_edge_loss_val)
            edge_lidar_loss_vec.append(cur_edge_lidar_loss_val)
            outputs.append(output)
            # Update progress bar if in rank 0
            if self.is_rank_0:
                # progress_bar.set_description(
                #     'Epoch {} | Avg.Loss {:.4f} | Sup.Loss {:.4f}'.format(
                #         module.current_epoch, output['loss'].item(),
                #         output['metrics']['supervised_loss'].item()))

                # progress_bar.set_description(
                #     'Epoch {} | Avg.Loss {:.4f} | Sup.Loss {:.4f} | Edge Loss {:.4f}'.format(
                #         module.current_epoch, output['loss'].item(),
                #         output['metrics']['supervised_loss'].item(),
                #         output['metrics']['edge_loss'].item()))

                cur_str = 'Epoch {} | Avg. {:.4f} | Sup. {:.4f} | Edge RGB {:.4f} | Edge Lidar {:.4f}'.format(
                        module.current_epoch, np.mean(all_loss_vec),
                        np.mean(supervised_loss_vec),
                        np.mean(edge_loss_vec),
                        np.mean(edge_lidar_loss_vec))
                if 'depth_loss' in output['metrics']:
                    depth_loss_val = output['metrics']['depth_loss'].item()
                    cur_str = cur_str + ' | Depth: ' + str(np.round(np.mean(depth_loss_val),2))
                if 'edge_from_depth_loss' in output['metrics']:
                    edge_from_depth_val = output['metrics']['edge_from_depth_loss'].item()
                    cur_str = cur_str + ' | Edge fD: ' + str(np.round(np.mean(edge_from_depth_val),2))
                if 'da_loss' in output['metrics']:
                    da_loss_val = output['metrics']['da_loss'].item()
                    cur_str = cur_str + ' | DA RGB: ' + str(np.round(np.mean(da_loss_val),4))
                if 'da_lidar_loss' in output['metrics']:
                    da_lidar_loss_val = output['metrics']['da_lidar_loss'].item()
                    cur_str = cur_str + ' | DA Lidar: ' + str(np.round(np.mean(da_lidar_loss_val),4))
                if 'normal_loss' in output['metrics']:
                    cur_normal_rgb_loss = output['metrics']['normal_loss'].item()
                    normal_loss_vec.append(cur_normal_rgb_loss)
                    cur_str = cur_str + ' | Normal RGB: ' + str(np.round(np.mean(normal_loss_vec),4))
                if 'normal_lidar_loss' in output['metrics']:
                    cur_normal_lidar_loss = output['metrics']['normal_lidar_loss'].item()
                    normal_lidar_loss_vec.append(cur_normal_lidar_loss)
                    cur_str = cur_str + ' | Normal Lidar: ' + str(np.round(np.mean(normal_lidar_loss_vec),4))

                progress_bar.set_description(cur_str)
        # Return outputs for epoch end
        return module.training_epoch_end(outputs)

    def validate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start validation loop
        all_outputs = []
        # For all validation datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.validation, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # if i > 10:
                #     break
                # Send batch to GPU and take a validation step
                batch = sample_to_cuda(batch)
                output = module.validation_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.validation_epoch_end(all_outputs)

    def test(self, module):
        # Send module to GPU
        module = module.to('cuda', dtype=self.dtype)
        # Get test dataloaders
        test_dataloaders = module.test_dataloader()
        # Run evaluation
        self.evaluate(test_dataloaders, module)

    @torch.no_grad()
    def evaluate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start evaluation loop
        all_outputs = []
        # For all test datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.test, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a test step
                batch = sample_to_cuda(batch, self.dtype)
                output = module.test_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.test_epoch_end(all_outputs)

    # def save_validation_graph(self, validation_keys, validation_values):
    #
    #     precision_idx = np.where([1 if 'precision' in validation_key else 0 for validation_key in validation_keys])[0]
    #     recall_idx = np.where([1 if 'recall' in validation_key else 0 for validation_key in validation_keys])[0]
    #
    #     input_precision_inner_idx = np.where([1 if 'input' in validation_key else 0 for validation_key in validation_keys[precision_idx]])[0]
    #     input_recall_inner_idx = np.where([1 if 'input' in validation_key else 0 for validation_key in validation_keys[recall_idx]])[0]
    #
    #     if len(input_precision_inner_idx) > 0:
    #         for i in range(0,len(input_precision_inner_idx)):
    #             plt.plot(validation_values[:,precision_idx[input_precision_inner_idx[i]]], validation_values[:,recall_idx[input_recall_inner_idx[i]]], 'o-', label='LIDAR input thresh ' + str(i))
    #
    #         precision_idx = np.delete(precision_idx, input_precision_inner_idx)
    #         recall_idx = np.delete(recall_idx, input_recall_inner_idx)
    #
    #     for i in range(0,len(precision_idx)):
    #         plt.plot(validation_values[:,precision_idx[i]], validation_values[:,recall_idx[i]], 'o-', label='RGB only thresh ' + str(i))
    #
    #     plt.title('Validation - edge precision to recall (approximate edge metric)')
    #     plt.xlabel('Precision')
    #     plt.ylabel('Recall')
    #     plt.legend()
    #     fig = plt.gcf()
    #     fig.set_size_inches(15, 10)
    #     # plt.show()
    #     fig.savefig(self.checkpoint.dirpath + '/validation_edge_graph.png')
    #     plt.close(fig)

