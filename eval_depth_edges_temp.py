# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import os
import matplotlib as mpl
import matplotlib.cm as cm
from itertools import product

from glob import glob
import matplotlib.pyplot as plt
import sys

from packnet_code.packnet_sfm.utils.tools import non_max_suppression
from eval_depth_edges import pr_evaluation
import glob

from packnet_code.packnet_sfm.utils.edge import edge_from_depth

def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM inference script')
    # parser.add_argument('file', type=str, help='Input file (.yaml)')
    # parser.add_argument('--checkpoint', type=str, help='Input file (.ckpt)')
    parser.add_argument('--config', type=str, help='Input file (.yaml)')
    args = parser.parse_args()
    # assert args.checkpoint.endswith(('.ckpt')), \
    #     'You need to provide a .ckpt file'
    assert args.config.endswith(('.yaml')), \
        'You need to provide a .yaml file'
    return args



def main(args):
    args = parse_args()

    edge_thresh_range = list(range(20, 241, 20))

    precision_vec,\
    recall_vec = pr_evaluation(config.edge_image_list,
                      args.save_folder + "/pred_list.txt",
                      config,
                      edge_thresh_range,
                      eval_mask_image_list=config.analysis.eval_mask_image_list)



if __name__ == '__main__':
    main(sys.argv[1:])
    # main(sys.argv[1:])

