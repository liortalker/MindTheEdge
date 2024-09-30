# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC), based on:
# https://github.com/TRI-ML/packnet-sfm - Toyota Research Institute

import glob
import numpy as np
import os
import cv2

from torch.utils.data import Dataset

from packnet_code.packnet_sfm.datasets.kitti_dataset_utils import \
    pose_from_oxts_packet, read_calib_file, transform_from_rot_trans
from packnet_code.packnet_sfm.utils.image import load_image
# from packnet_code.packnet_sfm.geometry.pose_utils import invert_pose_numpy

from packnet_code.packnet_sfm.datasets.kitti_dataset import read_png_depth

########################################################################################################################

# Cameras from the stero pair (left is the origin)
IMAGE_FOLDER = {
    'left': 'image_02',
    'right': 'image_03',
}
# Name of different calibration files
CALIB_FILE = {
    'cam2cam': 'calib_cam_to_cam.txt',
    'velo2cam': 'calib_velo_to_cam.txt',
    'imu2velo': 'calib_imu_to_velo.txt',
}
PNG_DEPTH_DATASETS = ['groundtruth']
OXTS_POSE_DATA = 'oxts'

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def read_lidar(filepath):
    """Reads in PointCloud from Kitti Dataset.
        Keyword Arguments:
        ------------------
        velo_dir : Str
                    Directory of the velodyne files.
        img_idx : Int
                  Index of the image.
        Returns:
        --------
        x : Numpy Array
                   Contains the x coordinates of the pointcloud.
        y : Numpy Array
                   Contains the y coordinates of the pointcloud.
        z : Numpy Array
                   Contains the z coordinates of the pointcloud.
        i : Numpy Array
                   Contains the intensity values of the pointcloud.
        [] : if file is not found
        """

    # if os.path.exists(filepath):
    with open(filepath, 'rb') as fid:
        data_array = np.fromfile(fid, np.single)

    xyzi = data_array.reshape(-1, 4)

    x = xyzi[:, 0]
    y = xyzi[:, 1]
    z = xyzi[:, 2]
    i = xyzi[:, 3]

    all_points = np.vstack((-y, -z, x)).T

    # filter points that have 0 intensity
    # all_points = all_points[i>0,:]

    # Remove nan points
    nan_mask = ~np.any(np.isnan(all_points), axis=1)
    point_cloud = all_points[nan_mask].T

    return point_cloud
    # else:
    #     return []


def process_lidar(raw_lidar_map, K, depth_map=None):
    lidar_mat = np.zeros((1080, 1920))
    # lidar_dim, num_lidar_pts = raw_lidar_map.shape
    # vio_count = 0
    p = np.matmul(K, raw_lidar_map)
    p_norm = p / p[2, :]
    in_range_idx = np.logical_and(
        np.logical_and(np.logical_and(p_norm[0, :] >= 0, p_norm[0, :] < 1920), p_norm[1, :] >= 0),
        p_norm[1, :] < 1080)
    p_norm = p_norm[:, in_range_idx].astype('int')
    p = p[:, in_range_idx]
    lidar_mat[p_norm[1, :], p_norm[0, :]] = p[2, :]

    if not depth_map is None:
        # remove lidar points with error more than 10 cm
        err_depth_vec = np.sqrt((lidar_mat - depth_map) ** 2)
        err_mask = (err_depth_vec > 0.1) * (lidar_mat > 0)
        lidar_mat[err_mask] = 0

    return lidar_mat

########################################################################################################################
#### DATASET
########################################################################################################################
import cv2

class GTADataset(Dataset):
    """
    KITTI dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    file_list : str
        Split file, with paths to the images to be used
    train : bool
        True if the dataset will be used for training
    data_transform : Function
        Transformations applied to the sample
    depth_type : str
        Which depth type to load
    with_pose : bool
        True if returning ground-truth pose
    back_context : int
        Number of backward frames to consider as context
    forward_context : int
        Number of forward frames to consider as context
    strides : tuple
        List of context strides
    """
    def __init__(self, root_dir, file_list, train=True,
                 data_transform=None, depth_type=None, input_depth_type=None,
                 with_pose=False, back_context=0, forward_context=0):
        # Assertions
        backward_context = back_context
        assert backward_context >= 0 and forward_context >= 0, 'Invalid contexts'

        self.backward_context = backward_context
        self.backward_context_paths = []
        self.forward_context = forward_context
        self.forward_context_paths = []

        self.with_context = (backward_context != 0 or forward_context != 0)
        self.split = file_list.split('/')[-1].split('.')[0]

        self.train = train
        self.root_dir = root_dir
        self.data_transform = data_transform

        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None
        self.with_pose = with_pose

        self.input_depth_type = input_depth_type
        self.with_input_depth = input_depth_type is not '' and input_depth_type is not None

        # self.depth_edges_all_scales = depth_edges_all_scales

        self._cache = {}
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}
        self.sequence_origin_cache = {}

        with open(file_list, "r") as f:
            data = f.readlines()

        self.image_paths = []
        self.depth_paths = []
        self.edge_paths = []
        self.lidar_paths = []
        self.seg_paths = []
        self.rgb_edge_paths = []
        self.rgb_edge_for_loss_paths = []
        self.normal_paths = []
        self.K = np.array([960, 0, 960, 0, 960, 540, 0, 0, 1]).reshape([3, 3])
        # Get file list from data
        for i, fname in enumerate(data):
            file_names = fname.split(' ')
            if file_names[-1] == '\n':
                file_names = file_names[0:-1]
            image_file_path = file_names[0].split('\n')[0]
            self.image_paths.append(image_file_path)
            if len(file_names) > 1:
                gt_file_path = file_names[1].split('\n')[0]
                self.depth_paths.append(gt_file_path)
            if len(file_names) > 2:
                edge_file_path = file_names[2].split('\n')[0]
                self.edge_paths.append(edge_file_path)
            if len(file_names) > 3:
                lidar_file_path = file_names[3].split('\n')[0]
                self.lidar_paths.append(lidar_file_path)
            if len(file_names) > 4:
                seg_file_path = file_names[4].split('\n')[0]
                if not seg_file_path == 'None':
                    self.seg_paths.append(seg_file_path)
            if len(file_names) > 5:
                rgb_edge_file_path = file_names[5].split('\n')[0]
                self.rgb_edge_paths.append(rgb_edge_file_path)
            if len(file_names) > 6:
                rgb_edge_for_loss_file_path = file_names[6].split('\n')[0]
                self.rgb_edge_for_loss_paths.append(rgb_edge_for_loss_file_path)
            if len(file_names) > 7:
                normal_file_path = file_names[7].split('\n')[0]
                self.normal_paths.append(normal_file_path)


########################################################################################################################

    @staticmethod
    def _get_next_file(idx, file):
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))
        return os.path.join(os.path.dirname(file), str(idx).zfill(len(base)) + ext)

    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file."""
        # return os.path.abspath(os.path.join(image_file, "../../../../.."))
        return os.path.abspath(os.path.join(image_file, "../../../.."))

    @staticmethod
    def _get_intrinsics(image_file, calib_data):
        """Get intrinsics from the calib_data dictionary."""
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return intrinsics
            if IMAGE_FOLDER[cam] in image_file:
                return np.reshape(calib_data[IMAGE_FOLDER[cam].replace('image', 'P_rect')], (3, 4))[:, :3]

    @staticmethod
    def _read_raw_calib_file(folder):
        """Read raw calibration files from folder."""
        return read_calib_file(os.path.join(folder, CALIB_FILE['cam2cam']))


    def _get_sample_context(self, sample_name,
                            backward_context, forward_context, stride=1):
        """
        Get a sample context

        Parameters
        ----------
        sample_name : str
            Path + Name of the sample
        backward_context : int
            Size of backward context
        forward_context : int
            Size of forward context
        stride : int
            Stride value to consider when building the context

        Returns
        -------
        backward_context : list of int
            List containing the indexes for the backward context
        forward_context : list of int
            List containing the indexes for the forward context
        """
        base, ext = os.path.splitext(os.path.basename(sample_name))
        parent_folder = os.path.dirname(sample_name)
        f_idx = int(base)

        # Check number of files in folder
        if parent_folder in self._cache:
            max_num_files = self._cache[parent_folder]
        else:
            max_num_files = len(glob.glob(os.path.join(parent_folder, '*' + ext)))
            self._cache[parent_folder] = max_num_files

        # Check bounds
        if (f_idx - backward_context * stride) < 0 or (
                f_idx + forward_context * stride) >= max_num_files:
            return None, None

        # Backward context
        c_idx = f_idx
        backward_context_idxs = []
        while len(backward_context_idxs) < backward_context and c_idx > 0:
            c_idx -= stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                backward_context_idxs.append(c_idx)
        if c_idx < 0:
            return None, None

        # Forward context
        c_idx = f_idx
        forward_context_idxs = []
        while len(forward_context_idxs) < forward_context and c_idx < max_num_files:
            c_idx += stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)
        if c_idx >= max_num_files:
            return None, None

        return backward_context_idxs, forward_context_idxs

    def _get_context_files(self, sample_name, idxs):
        """
        Returns image and depth context files

        Parameters
        ----------
        sample_name : str
            Name of current sample
        idxs : list of idxs
            Context indexes

        Returns
        -------
        image_context_paths : list of str
            List of image names for the context
        depth_context_paths : list of str
            List of depth names for the context
        """
        image_context_paths = [self._get_next_file(i, sample_name) for i in idxs]
        return image_context_paths, None



    def __len__(self):
        """Dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""
        # print(idx)
        # idx = 1
        # Add image information
        sample = {
            'idx': idx,
            'filename': '%s_%010d' % (self.split, idx),
            'rgb': load_image(self.image_paths[idx])
        }

        if hasattr(self, 'depth_paths'):
            # sample['depth'] = np.load(self.depth_paths[idx])
            if self.depth_paths[idx].split('.')[-1] == 'png':
                sample['depth'] = read_png_depth(self.depth_paths[idx])

            elif self.depth_paths[idx].split('.')[-1] == 'bin':
                sample['depth'] = self.depth_read_bin(self.depth_paths[idx])

            elif self.depth_paths[idx].split('.')[-1] == 'npy':
                sample['depth'] = np.load(self.depth_paths[idx])

        if hasattr(self, 'edge_paths'):
            import cv2
            if not self.edge_paths[idx] == 'None':
                if self.edge_paths[idx].split('.')[-1] == 'png':
                    sample['edge'] = cv2.imread(self.edge_paths[idx])[:,:,0]
                elif self.edge_paths[idx].split('.')[-1] == 'npy':
                    sample['edge'] = np.load(self.edge_paths[idx])
                # if self.depth_edges_all_scales:
                # check if you have files for scales > 0
                scale_more_than_0_filename = self.edge_paths[idx].split('_000')[0] + '_001.png'
                if os.path.exists(scale_more_than_0_filename):
                    for i in range(1,4):
                        sample['edge_' + str(i)] = cv2.imread(self.edge_paths[idx].split('_000')[0] + '_00' + str(i) + '.png')[:, :, 0]

        if hasattr(self, 'lidar_paths'):

            if self.lidar_paths[idx].split('.')[-1] == 'png':
                lidar_mat = read_png_depth(self.lidar_paths[idx])

            elif self.lidar_paths[idx].split('.')[-1] == 'bin':
                raw_lidar_map = read_lidar(self.lidar_paths[idx])
                lidar_mat = process_lidar(raw_lidar_map, self.K, sample['depth'])

            elif self.lidar_paths[idx].split('.')[-1] == 'npy':
                lidar_mat = np.load(self.lidar_paths[idx])

            if not self.input_depth_type == '' and not self.input_depth_type is None:
                sample['input_depth'] = lidar_mat
            sample['lidar'] = lidar_mat

        if hasattr(self, 'seg_paths'):
            if len(self.seg_paths) > 0:
                # sample['seg'] = load_image(self.seg_paths[idx])
                sample['seg'] = cv2.imread(self.seg_paths[idx])
                sample['seg'] = cv2.cvtColor(sample['seg'], cv2.COLOR_BGR2RGB)

        if hasattr(self, 'rgb_edge_paths'):
            if len(self.rgb_edge_paths) > 0:
                if not self.rgb_edge_paths[0] == 'None' and not self.rgb_edge_paths[0] is None:
                    if self.rgb_edge_paths[idx].split('.')[-1] == 'png':
                        sample['rgb_edge'] = cv2.imread(self.rgb_edge_paths[idx])[:, :, 0]
                    elif self.rgb_edge_paths[idx].split('.')[-1] == 'npy':
                        sample['rgb_edge'] = np.load(self.rgb_edge_paths[idx])

        if hasattr(self, 'rgb_edge_for_loss_paths'):
            if len(self.rgb_edge_for_loss_paths) > 0:
                if not self.rgb_edge_for_loss_paths[0] == 'None' and not self.rgb_edge_for_loss_paths[0] is None:
                    if self.rgb_edge_for_loss_paths[idx].split('.')[-1] == 'png':
                        sample['rgb_edge_for_loss'] = cv2.imread(self.rgb_edge_for_loss_paths[idx])[:, :, 0]
                    elif self.rgb_edge_for_loss_paths[idx].split('.')[-1] == 'npy':
                        sample['rgb_edge_for_loss'] = np.load(self.rgb_edge_for_loss_paths[idx])

        if hasattr(self, 'normal_paths'):
            if len(self.normal_paths) > 0:
                if not self.normal_paths[0] == 'None' and not self.normal_paths[0] is None:
                    if self.normal_paths[idx].split('.')[-1] == 'png':
                        sample['normal'] = cv2.imread(self.normal_paths[idx])[:, :, 0]
                        # mapping from image uint8 to arctan2 values
                        # sobel_angle_255 = (((sobel_angle*(180/np.pi)+180)/360)*255).astype('uint8')
                        sample['normal'] = (360.*(sample['normal']/255.) - 180)*(np.pi/180)

                        # if self.depth_edges_all_scales:
                        # check if you have files for scales > 0
                        scale_more_than_0_filename = self.normal_paths[idx].split('_000')[0] + '_001.png'
                        if os.path.exists(scale_more_than_0_filename):
                            for i in range(1, 4):
                                sample['normal_' + str(i)] = cv2.imread(
                                    self.normal_paths[idx].split('_000')[0] + '_00' + str(i) + '.png')[:, :, 0]
                                sample['normal_' + str(i)] = (360. * (sample['normal_' + str(i)] / 255.) - 180) * (np.pi / 180)

        # Apply transformations
        if self.data_transform:
            sample = self.data_transform(sample)

        # Return sample
        return sample

    def ndcToDepth_vec(self, ndc):
        nc_z = 0.15
        fc_z = 600

        rows, cols = ndc.shape
        nc_z_mat = np.ones((rows, cols)) * nc_z
        d_nc = nc_z_mat

        depth = d_nc / (ndc + (nc_z_mat * d_nc / (2 * fc_z)))
        depth[ndc == 0.0] = fc_z

        return depth

    def depth_read_bin(self, filename, rows=1080, cols=1920):

        fd = open(filename.split('\n')[0], 'rb')
        f = np.fromfile(fd, dtype=np.float32, count=rows*cols)
        ndc = f.reshape((rows, cols))

        depth = self.ndcToDepth_vec(ndc)

        return depth



########################################################################################################################
