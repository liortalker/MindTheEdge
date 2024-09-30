"""Default packnet_sfm configuration parameters (overridable in configs/*.yaml)
"""

import os
from yacs.config import CfgNode as CN

########################################################################################################################
cfg = CN()
cfg.name = ''       # Run name
cfg.debug = False   # Debugging flag
cfg.is_multi_gpu = False # to run on multiple gpus
########################################################################################################################
### ARCH
########################################################################################################################
cfg.arch = CN()
cfg.arch.seed = 42                      # Random seed for Pytorch/Numpy initialization
cfg.arch.min_epochs = 1                 # Minimum number of epochs
cfg.arch.max_epochs = 51                # Maximum number of epochs
cfg.arch.validate_first = False         # Validate before training starts
########################################################################################################################
### CHECKPOINT
########################################################################################################################
cfg.checkpoint = CN()
cfg.checkpoint.filepath = ''            # Checkpoint filepath to save data
cfg.checkpoint.save_top_k = 5           # Number of best models to save
cfg.checkpoint.monitor = 'loss'         # Metric to monitor for logging
cfg.checkpoint.monitor_index = 0        # Dataset index for the metric to monitor
cfg.checkpoint.mode = 'auto'            # Automatically determine direction of improvement (increase or decrease)
cfg.checkpoint.s3_path = ''             # s3 path for AWS model syncing
cfg.checkpoint.s3_frequency = 1         # How often to s3 sync
cfg.checkpoint.save_freq = 5            # How often to save the model
cfg.checkpoint.yaml_path = ''            # placeholder to yaml filepath for saving in dir
########################################################################################################################
### SAVE
########################################################################################################################
cfg.save = CN()
cfg.save.folder = ''                    # Folder where data will be saved
cfg.save.depth = CN()
cfg.save.depth.rgb = True               # Flag for saving rgb images
cfg.save.depth.viz = True               # Flag for saving inverse depth map visualization
cfg.save.depth.npz = True               # Flag for saving numpy depth maps
cfg.save.depth.png = True               # Flag for saving png depth maps
cfg.save.depth.multiscale = False       # Infer multiscale or not
########################################################################################################################
### WANDB
########################################################################################################################
cfg.wandb = CN()
cfg.wandb.dry_run = True                                 # Wandb dry-run (not logging)
cfg.wandb.name = ''                                      # Wandb run name
cfg.wandb.project = os.environ.get("WANDB_PROJECT", "")  # Wandb project
cfg.wandb.entity = os.environ.get("WANDB_ENTITY", "")    # Wandb entity
cfg.wandb.tags = []                                      # Wandb tags
cfg.wandb.dir = ''                                       # Wandb save folder
cfg.wandb.train_log_step = 50                            # Number of training iterations to update graphs
########################################################################################################################
### MODEL
########################################################################################################################
cfg.model = CN()
cfg.model.name = ''                         # Training model
cfg.model.checkpoint_path = ''              # Checkpoint path for model saving
########################################################################################################################
### MODEL.OPTIMIZER
########################################################################################################################
cfg.model.optimizer = CN()
cfg.model.optimizer.name = 'Adam'               # Optimizer name
cfg.model.optimizer.depth = CN()
cfg.model.optimizer.depth.lr = 0.0002           # Depth learning rate
cfg.model.optimizer.depth.weight_decay = 0.0    # Dept weight decay
cfg.model.optimizer.pose = CN()
cfg.model.optimizer.pose.lr = 0.0002            # Pose learning rate
cfg.model.optimizer.pose.weight_decay = 0.0     # Pose weight decay
########################################################################################################################
### MODEL.SCHEDULER
########################################################################################################################
cfg.model.scheduler = CN()
cfg.model.scheduler.name = 'StepLR'     # Scheduler name
cfg.model.scheduler.step_size = 10      # Scheduler step size
cfg.model.scheduler.gamma = 0.5         # Scheduler gamma value
cfg.model.scheduler.T_max = 20          # Scheduler maximum number of iterations
########################################################################################################################
### MODEL.PARAMS
########################################################################################################################
cfg.model.params = CN()
cfg.model.params.crop = ''                # Which crop should be used during evaluation
cfg.model.params.min_depth = 0.0          # Minimum depth value to evaluate
cfg.model.params.max_depth = 80.0         # Maximum depth value to evaluate
cfg.model.params.scale_output = 'resize'  # Depth resizing function
########################################################################################################################
### MODEL.LOSS
########################################################################################################################
cfg.model.loss = CN()
#
cfg.model.loss.num_scales = 4                   # Number of inverse depth scales to use
cfg.model.loss.progressive_scaling = 0.0        # Training percentage to decay number of scales
cfg.model.loss.flip_lr_prob = 0.5               # Probablity of horizontal flippping
cfg.model.loss.rotation_mode = 'euler'          # Rotation mode
cfg.model.loss.upsample_depth_maps = True       # Resize depth maps to highest resolution
#
cfg.model.loss.ssim_loss_weight = 0.85          # SSIM loss weight
cfg.model.loss.occ_reg_weight = 0.1             # Occlusion regularizer loss weight
cfg.model.loss.smooth_loss_weight = 0.001       # Smoothness loss weight
cfg.model.loss.C1 = 1e-4                        # SSIM parameter
cfg.model.loss.C2 = 9e-4                        # SSIM parameter
cfg.model.loss.photometric_reduce_op = 'min'    # Method for photometric loss reducing
cfg.model.loss.disp_norm = True                 # Inverse depth normalization
cfg.model.loss.clip_loss = 0.0                  # Clip loss threshold variance
cfg.model.loss.padding_mode = 'zeros'           # Photometric loss padding mode
cfg.model.loss.automask_loss = True             # Automasking to remove static pixels
#
cfg.model.loss.velocity_loss_weight = 0.1       # Velocity supervision loss weight
#
cfg.model.loss.supervised_method = 'sparse-l1'  # Method for depth supervision
cfg.model.loss.supervised_num_scales = 4        # Number of scales for supervised learning
cfg.model.loss.supervised_loss_weight = 0.9     # Supervised loss weight
cfg.model.loss.depth_edges_loss_weight = 10.0   # The weight for the edges loss
cfg.model.loss.edges_depth_edge_loss_all_scales = False # apply edge loss on all scales? (4 total scales)
cfg.model.loss.edges_is_da_on_features = False  # apply discriminator on the features (on bottleneck layer)
cfg.model.loss.edges_multi_layer_da_on_features = True  # apply features da on multiple layers
cfg.model.loss.edges_is_da_on_output = False    # apply discriminator on the output
########################################################################################################################
### MODEL.EDGES
########################################################################################################################
cfg.edges = CN()
cfg.edges.train_depth_edges = False
cfg.edges.depth_edges_loss_weight = 10.0
cfg.edges.depth_edge_loss_pos_to_neg_weight = 1.0
cfg.edges.depth_edges_images_log = False
cfg.edges.depth_edges_metric_log = False
cfg.edges.fixed_training_seed_sequence = []
cfg.edges.edge_loss_type = 'cross_entropy'
cfg.edges.source_target_equal_weight_loss = False
cfg.edges.idx_example_to_overfit = -1
cfg.edges.use_external_edges_for_loss = True  # if False edges will be computed by canny from the GT
cfg.edges.edge_loss_class_list_to_mask_out = []
########################################################################################################################
### MODEL.DEPTH_NET
########################################################################################################################
cfg.model.depth_net = CN()
cfg.model.depth_net.name = ''               # Depth network name
cfg.model.depth_net.checkpoint_path = ''    # Depth checkpoint filepath
cfg.model.depth_net.version = ''            # Depth network version
cfg.model.depth_net.dropout = 0.0           # Depth network dropout
cfg.model.depth_net.freeze_encoder = False   # Freeze the weights of the depth net encoder
cfg.model.depth_net.freeze_decoder = False   # Freeze the weights of the depth net decoder
cfg.model.depth_net.freeze_san = False   # Freeze the weights of the depth net san
cfg.model.depth_net.input_channels = 3   # num of input channels. 3 for RGB, 4 for RGB + RGB edges
cfg.model.depth_net.is_depth_aux_net = False # use depth auxilary net as another loss for edges
cfg.model.depth_net.output_channels = 1  # num of output channels. 1 for edges/depth. 2 for edges/depth + normals
########################################################################################################################
### MODEL.POSE_NET
########################################################################################################################
cfg.model.pose_net = CN()
cfg.model.pose_net.name = ''                # Pose network name
cfg.model.pose_net.checkpoint_path = ''     # Pose checkpoint filepath
cfg.model.pose_net.version = ''             # Pose network version
cfg.model.pose_net.dropout = 0.0            # Pose network dropout
########################################################################################################################
### DATASETS
########################################################################################################################
cfg.datasets = CN()
########################################################################################################################
### DATASETS.AUGMENTATION
########################################################################################################################
cfg.datasets.augmentation = CN()
cfg.datasets.augmentation.image_shape = ()                      # Image shape
cfg.datasets.augmentation.jittering = (0.2, 0.2, 0.2, 0.05)     # Color jittering values
cfg.datasets.augmentation.crop_train_borders = ()               # Crop training borders
cfg.datasets.augmentation.crop_eval_borders = ()                # Crop evaluation borders
cfg.datasets.augmentation.lidar_scale = ()                      # Augment the lidar input by scaling in X or 1/X
cfg.datasets.augmentation.lidar_add = ()                        # Augment the lidar input by translating in X
cfg.datasets.augmentation.lidar_drop_rate = 0.0                 # Maximal percentage of lidar points to randomly drop
########################################################################################################################
### DATASETS.TRAIN
########################################################################################################################
cfg.datasets.train = CN()
cfg.datasets.train.batch_size = 8                   # Training batch size
cfg.datasets.train.num_workers = 16                 # Training number of workers
cfg.datasets.train.back_context = 1                 # Training backward context
cfg.datasets.train.forward_context = 1              # Training forward context
cfg.datasets.train.dataset = []                     # Training dataset
cfg.datasets.train.path = []                        # Training data path
cfg.datasets.train.split = []                       # Training split
cfg.datasets.train.depth_type = ['']                # Training depth type
cfg.datasets.train.input_depth_type = ['']          # Training input depth type
cfg.datasets.train.cameras = [[]]                   # Training cameras (double list, one for each dataset)
cfg.datasets.train.repeat = [1]                     # Number of times training dataset is repeated per epoch
cfg.datasets.train.num_logs = 5                     # Number of training images to log

########################################################################################################################
### DATASETS.VALIDATION
########################################################################################################################
cfg.datasets.validation = CN()
cfg.datasets.validation.batch_size = 1              # Validation batch size
cfg.datasets.validation.num_workers = 8             # Validation number of workers
cfg.datasets.validation.back_context = 0            # Validation backward context
cfg.datasets.validation.forward_context = 0         # Validation forward contxt
cfg.datasets.validation.dataset = []                # Validation dataset
cfg.datasets.validation.path = []                   # Validation data path
cfg.datasets.validation.split = []                  # Validation split
cfg.datasets.validation.depth_type = ['']           # Validation depth type
cfg.datasets.validation.input_depth_type = ['']     # Validation input depth type
cfg.datasets.validation.cameras = [[]]              # Validation cameras (double list, one for each dataset)
cfg.datasets.validation.num_logs = 5                # Number of validation images to log
cfg.datasets.validation.gt_crop = []                # Crop the images from dataset X [start_X, end_X, start_Y, end_Y] in absolute pixels
########################################################################################################################
### DATASETS.TEST
########################################################################################################################
cfg.datasets.test = CN()
cfg.datasets.test.batch_size = 1                    # Test batch size
cfg.datasets.test.num_workers = 8                   # Test number of workers
cfg.datasets.test.back_context = 0                  # Test backward context
cfg.datasets.test.forward_context = 0               # Test forward context
cfg.datasets.test.dataset = []                      # Test dataset
cfg.datasets.test.path = []                         # Test data path
cfg.datasets.test.split = []                        # Test split
cfg.datasets.test.depth_type = ['']                 # Test depth type
cfg.datasets.test.input_depth_type = ['']           # Test input depth type
cfg.datasets.test.cameras = [[]]                    # Test cameras (double list, one for each dataset)
cfg.datasets.test.num_logs = 5                      # Number of test images to log
cfg.datasets.test.nms = False                       # To use non-maximum suppresion or not
cfg.datasets.test.hysteresis = False                # To use hysteresis (like in Canny edge detector)
cfg.datasets.test.normals = False                   # To compute normals in edge estimation inference
cfg.datasets.test.is_infer_rgb = True               # To do inference for RGB only
cfg.datasets.test.is_infer_lidar = True             # To do inference for RGB + lidar

########################################################################################################################
####### ANALYSIS
########################################################################################################################
cfg.analysis = CN()
cfg.analysis.just_evaluate = False
cfg.analysis.run_metrics = False
cfg.analysis.run_light_edge_metrics = False
cfg.analysis.run_heavy_edge_metrics = False

cfg.analysis.save_error_plot = False

cfg.analysis.gt_image_list = ""
cfg.analysis.edge_image_list = ""
cfg.analysis.eval_mask_image_list = ""

cfg.analysis.type = "dense"
cfg.analysis.shape = False

cfg.analysis.intrinsics = False #[ [ 1158, 0, 620.5],[ 0, 1158, 188],[ 0, 0, 1] ]
cfg.analysis.distortion_params = False

cfg.analysis.start_frm_idx = 0
cfg.analysis.end_frm_idx = -1
cfg.analysis.min_depth = 0.01
cfg.analysis.max_depth = 80.

cfg.analysis.prec_recall_eval_range_min = 0.12
cfg.analysis.prec_recall_eval_range_max = 0.65

# cfg.analysis.gt_crop = [0, 1, 0, 1]
cfg.analysis.gt_crop = []
cfg.analysis.gt_type = "depth"
cfg.analysis.rel_err_lo = -1
cfg.analysis.rel_err_hi = 10.
cfg.analysis.hist_num_bins = 300
cfg.analysis.out_file_name = "analyzer_data.pkl",
cfg.analysis.median_scaling = "median_of_fractions"

  # Crop used by Garg ECCV16 to reproduce Eigen NIPS14 results.
  # If used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]

cfg.analysis.gt_crop = [0, 1, 0, 1]

cfg.analysis.mask_epipole = False
cfg.analysis.epipole_mask_radius = -1

########################################################################################################################
####### VISUALIZATION
########################################################################################################################
cfg.visualization = CN()
cfg.visualization.online_vis = False
cfg.visualization.offline_vis = False


########################################################################################################################
### THESE SHOULD NOT BE CHANGED
########################################################################################################################
cfg.config = ''                 # Run configuration file
cfg.default = ''                # Run default configuration file
cfg.wandb.url = ''              # Wandb URL
cfg.checkpoint.s3_url = ''      # s3 URL
cfg.save.pretrained = ''        # Pretrained checkpoint
cfg.prepared = False            # Prepared flag
########################################################################################################################

def get_cfg_defaults():
    return cfg.clone()