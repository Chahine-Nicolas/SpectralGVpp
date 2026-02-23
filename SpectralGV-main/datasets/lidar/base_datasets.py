import os
import sys
import pickle
from typing import List, Dict
import torch
import numpy as np
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.alita.alita_raw import ALITAPointCloudLoader
from datasets.kitti.kitti_raw import KittiPointCloudLoader
from datasets.mulran.mulran_raw import MulranPointCloudLoader
from datasets.southbay.southbay_raw import SouthbayPointCloudLoader
from datasets.kitti360.kitti360_raw import Kitti360PointCloudLoader
from datasets.point_clouds_utils import PointCloudLoader
from datasets.lidar.lidar_raw import LidarPointCloudLoader

class TrainingTuple:
    """
    Tuple describing an element for training/validation.

    Attributes:
        id (int): Element ID (IDs start from 0 and are consecutive numbers).
        timestamp (int): Timestamp of the scan.
        rel_scan_filepath (str): Relative path to the scan file.
        positives (np.ndarray): Sorted array of positive element IDs.
        non_negatives (np.ndarray): Sorted array of non-negative element IDs.
        pose (np.ndarray): 4x4 pose matrix.
        positives_poses (Dict[int, np.ndarray], optional): Relative poses of positive examples refined using ICP.
    """
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, pose: np.ndarray, positives_poses: Dict[int, np.ndarray] = None):
        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.pose = pose
        self.positives_poses = positives_poses

class EvaluationTuple:
    """
    Tuple describing an evaluation set element.

    Attributes:
        timestamp (int): Timestamp of the scan.
        rel_scan_filepath (str): Relative path to the scan file.
        position (np.ndarray): (x, y) position in meters.
        pose (np.ndarray, optional): 6 DoF pose as 4x4 pose matrix.
    """
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.ndarray, pose: np.ndarray = None):
        assert position.shape == (2,)
        assert pose is None or pose.shape == (4, 4)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position
        self.pose = pose

    def to_tuple(self) -> tuple:
        """
        Converts the EvaluationTuple to a tuple for serialization.

        Returns:
            tuple: (timestamp, rel_scan_filepath, position, pose)
        """
        return self.timestamp, self.rel_scan_filepath, self.position, self.pose

class TrainingDataset(Dataset):
    """
    Dataset class for training, handling point cloud loading and indexing.

    Attributes:
        dataset_path (str): Path to the dataset.
        dataset_type (str): Type of the dataset (e.g., 'mulran', 'southbay').
        query_filepath (str): Path to the query file.
        transform (callable, optional): Optional transform to be applied on a sample.
        set_transform (callable, optional): Optional transform to be applied on the set.
        queries (Dict[int, TrainingTuple]): Dictionary of training tuples.
        pc_loader (PointCloudLoader): Point cloud loader for the dataset type.
    """
    def __init__(self, dataset_path: str, dataset_type: str, query_filename: str, transform=None, set_transform=None):
        assert os.path.exists(dataset_path), f'Cannot access dataset path: {dataset_path}'
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), f'Cannot access query file: {self.query_filepath}'
        self.transform = transform
        self.set_transform = set_transform
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        print(f'{len(self)} queries in the dataset')
        self.pc_loader = get_pointcloud_loader(self.dataset_type)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, ndx: int) -> tuple:
        """
        Loads and returns a point cloud and its index.

        Args:
            ndx (int): Index of the query.

        Returns:
            tuple: (query_pc, ndx)
        """
        file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
        query_pc = self.pc_loader(file_pathname)
        query_pc = torch.tensor(query_pc, dtype=torch.float)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        return query_pc, ndx

    def get_positives(self, ndx: int) -> np.ndarray:
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx: int) -> np.ndarray:
        return self.queries[ndx].non_negatives

class EvaluationSet:
    """
    Evaluation set consisting of map and query elements.

    Attributes:
        query_set (List[EvaluationTuple]): List of query elements.
        map_set (List[EvaluationTuple]): List of map elements.
    """
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str) -> None:
        """
        Saves the evaluation set to a pickle file.

        Args:
            pickle_filepath (str): Path to the pickle file.
        """
        query_l = [e.to_tuple() for e in self.query_set]
        map_l = [e.to_tuple() for e in self.map_set]
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str) -> None:
        """
        Loads the evaluation set from a pickle file.

        Args:
            pickle_filepath (str): Path to the pickle file.
        """
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))
        self.query_set = [EvaluationTuple(e[0], e[1], e[2], e[3]) for e in query_l]
        self.map_set = [EvaluationTuple(e[0], e[1], e[2], e[3]) for e in map_l]

    def get_map_positions(self) -> np.ndarray:
        """
        Returns the map positions as a (N, 2) array.

        Returns:
            np.ndarray: Array of map positions.
        """
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self) -> np.ndarray:
        """
        Returns the query positions as a (N, 2) array.

        Returns:
            np.ndarray: Array of query positions.
        """
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions

def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple], dist_threshold: float) -> List[EvaluationTuple]:
    """
    Filters out query elements without a corresponding map element within a distance threshold.

    Args:
        query_set (List[EvaluationTuple]): List of query elements.
        map_set (List[EvaluationTuple]): List of map elements.
        dist_threshold (float): Distance threshold in meters.

    Returns:
        List[EvaluationTuple]: Filtered list of query elements.
    """
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position
    kdtree = KDTree(map_pos)
    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1
    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] radius")
    return filtered_query_set

def get_pointcloud_loader(dataset_type: str) -> PointCloudLoader:
    """
    Returns the appropriate point cloud loader for the given dataset type.

    Args:
        dataset_type (str): Type of the dataset.

    Returns:
        PointCloudLoader: Point cloud loader for the dataset type.

    Raises:
        NotImplementedError: If the dataset type is not supported.
    """
    if dataset_type == 'mulran':
        return MulranPointCloudLoader()
    elif dataset_type == 'southbay':
        return SouthbayPointCloudLoader()
    elif dataset_type == 'kitti':
        return KittiPointCloudLoader()
    elif dataset_type == 'alita':
        return ALITAPointCloudLoader()
    elif dataset_type == 'kitti360':
        return Kitti360PointCloudLoader()
    elif dataset_type == 'lidar':
        return LidarPointCloudLoader()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")
