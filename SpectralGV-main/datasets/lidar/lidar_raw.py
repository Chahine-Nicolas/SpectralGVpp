import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from datasets.point_clouds_utils import PointCloudLoader

class LidarPointCloudLoader(PointCloudLoader):
    """Extends PointCloudLoader for LiDAR-specific point cloud operations."""

    def set_properties(self):
        """
        Sets default properties for LiDAR point clouds.

        Args: None
        Returns: None
        Sets:
            ground_plane_level (float): Default ground plane level (-1.5 meters).
        """
        self.ground_plane_level = -1.5

    def read_pc(self, file_pathname: str) -> torch.Tensor:
        """
        Reads a LiDAR point cloud file without preprocessing.

        Args:
            file_pathname (str): Path to the point cloud file.

        Returns:
            torch.Tensor: Nx4 tensor containing x,y,z,reflectance values.
        """
        pc = np.fromfile(file_pathname, dtype=np.float32)
        pc = np.reshape(pc, (-1, 4))  # [num_points, 4] -> x,y,z,reflectance
        return pc

class LidarSequence(Dataset):
    """
    Represents a sequence of LiDAR point clouds from a raw dataset.
    Handles loading, pose information, and timestamp management.
    """

    def __init__(self, dataset_root: str, rel_lidar_path: str):
        """
        Initializes a LiDAR sequence from the dataset.

        Args:
            dataset_root (str): Root directory of the dataset.
            rel_lidar_path (str): Relative path to LiDAR scans.

        Attributes:
            rel_lidar_timestamps (list): List of timestamps for each scan.
            lidar_poses (np.ndarray): 4x4 pose matrices for each scan.
            rel_scan_filepath (list): List of scan file paths.
            dictionary_id (dict): Mapping of scan filenames to indices.
        """
        print(dataset_root)
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        self.dataset_root = dataset_root
        self.rel_lidar_path = rel_lidar_path
        self.rel_lidar_timestamps, self.lidar_poses,  self.rel_scan_filepath, self.dictionary_id = self._read_lidar_poses()
        #self.rel_scan_filepath = [os.path.join(self.rel_lidar_path, (e + '.bin')) for e in filenames]

    def __len__(self):
        """
        Returns the number of scans in the sequence.

        Returns:
            int: Number of scans.
        """
        return len(self.rel_lidar_timestamps)

    def __getitem__(self, ndx: int) -> dict:
        """
        Retrieves a scan and its metadata by index.

        Args:
            ndx (int): Index of the scan to retrieve.

        Returns:
            dict: Dictionary containing:
                - 'pc' (np.ndarray): Point cloud data.
                - 'pose' (np.ndarray): 4x4 pose matrix.
                - 'ts' (int): Timestamp.
        """
        scan_filepath = os.path.join(self.dataset_root, self.rel_scan_filepath[ndx])
        pc = load_pc(scan_filepath)
        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]
        return {'pc': pc, 'pose': self.lidar_poses[ndx], 'ts': self.rel_lidar_timestamps[ndx]}

    def _read_lidar_poses(self):
        """
        Reads LiDAR scan filenames and generates pose information.

        Args: None

        Returns:
            tuple: Contains:
                - rel_ts (list): List of timestamps.
                - poses (np.ndarray): Array of 4x4 pose matrices.
                - filenames (list): List of scan filenames.
                - dictionary_id (dict): Mapping of filenames to indices.
        """
        dictionary_id = {}
        rel_ts = []
        filenames = []
        ndx = 0
        decoupage = 20
        fnames = os.listdir(os.path.join(self.dataset_root, self.rel_lidar_path))
        temp = os.path.join(self.dataset_root, self.rel_lidar_path)
        poses_dir = os.path.join(self.dataset_root, 'dictionaire')
        if not os.path.exists(poses_dir):
            os.makedirs(poses_dir)
        poses = np.zeros((len(fnames), 4, 4), dtype=np.float64)

        for e in fnames:
            if os.path.isfile(os.path.join(temp, e)):                
                filenames.append(os.path.join(self.rel_lidar_path,os.path.splitext(e)[0] + '.bin'))                
                liste = e.split("_")
                dictionary_id[os.path.splitext(e)[0]] = ndx
                pose_x = (int(liste[2])-656)*1000 + int(liste[8])*decoupage+decoupage/2
                pose_y = (int(liste[3])-6860)*1000 + int(liste[9])*decoupage+decoupage/2
                poses[ndx] = np.array([[1, 0, 0, pose_x],
                                       [0, 1, 0, pose_y],
                                       [0, 0, 1, 100],
                                       [0., 0., 0., 1.]])
                rel_ts.append(ndx)
                ndx += 1


        json_file_path = os.path.join(poses_dir, f'dictionaire_neuf_zone.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(dictionary_id, json_file, indent=4)

        assert len(fnames) > 0, f"Make sure that the path {self.rel_lidar_path} is correct"
        #filenames = [os.path.split(fnames_chemain)[-1][:-4] for fnames_chemain in fnames_chemain]
        return rel_ts, poses, filenames, dictionary_id

def load_pc(filepath: str) -> np.ndarray:
    """
    Loads a point cloud from a file without applying any transformations.

    Args:
        filepath (str): Path to the point cloud file.

    Returns:
        np.ndarray: Nx4 array containing x,y,z,reflectance values.
    """
    pc = np.fromfile(filepath, dtype=np.float32)
    pc = np.reshape(pc, (-1, 4))  # [num_points, 4] -> x,y,z,reflectance
    return pc

