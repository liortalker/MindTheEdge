# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC), based on:
# https://github.com/TRI-ML/packnet-sfm - Toyota Research Institute

import argparse
import sys

from packnet_code.packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_code.packnet_sfm.models.model_checkpoint import ModelCheckpoint
from packnet_code.packnet_sfm.trainers.common_trainer import CommonTrainer
from packnet_code.packnet_sfm.utils.config import parse_train_file
from packnet_code.packnet_sfm.utils.load import set_debug, filter_args_create
from packnet_code.packnet_sfm.utils.horovod import hvd_init, rank
from packnet_code.packnet_sfm.loggers.wandb_logger import WandbLogger


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM training script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args()
    # assert args.file.endswith(('.ckpt', '.yaml')), \
    #     'You need to provide a .ckpt of .yaml file'
    return args


def train(file):
    """
    Monocular depth estimation training script.

    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    # Initialize horovod
    # hvd_init()

    # Produce configuration and checkpoint from filename
    config, ckpt = parse_train_file(file)

    # Set debug if requested
    set_debug(config.debug)

    # Wandb Logger
    logger = None if config.wandb.dry_run or rank() > 0 \
        else filter_args_create(WandbLogger, config.wandb)

    # model checkpoint
    checkpoint = None if config.checkpoint.filepath is '' or rank() > 0 else \
        filter_args_create(ModelCheckpoint, config.checkpoint)

    # Initialize model wrapper
    #print('Before model wrapper')
    model_wrapper = ModelWrapper(config, resume=ckpt, logger=logger)
    # if config.is_multi_gpu:
    #     model_wrapper = torch.nn.DataParallel(model_wrapper)

    # Create trainer with args.arch parameters
    trainer = CommonTrainer(**config.arch, checkpoint=checkpoint)

    # Train model
    trainer.fit(model_wrapper)

def main(args):
    args = parse_args()
    train(args.file)

if __name__ == '__main__':
    main(sys.argv[1:])
    # main(sys.argv[1:])