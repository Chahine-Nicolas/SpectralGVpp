#import glob
import argparse
import open3d as o3d
import tensorflow as tf
import numpy as np
import os
import copy
import sys
import time
from utils.config import Config
from datasets.common import Dataset
from models.KPFCNN_model import KernelPointFCNN
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
import pickle
from tqdm import tqdm
from typing import List
from pathlib import Path
import shutil


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"


class MiniDataset(Dataset):
    """
    A class to create and manage a mini point cloud dataset.
    """

    def __init__(self, files, retranche, voxel_size=0.03):
        """
        Initializes the MiniDataset by loading and splitting point clouds.

        Args:
            files (list): List of paths to point cloud files.
            retranche (int): Number of points to extract per iteration.
            voxel_size (float, optional): Voxel size for downsampling. Defaults to 0.03.

        Attributes:
            anc_points (dict): Dictionary containing split point cloud data.
            ids_list (dict): Dictionary mapping split data to filenames.
            num_test (int): Total number of subsets created.
        """
        Dataset.__init__(self, 'Mini')
        self.num_test = 0
        self.anc_points = {"train": [], "test": []}
        self.ids_list = {"train": [], "test": []}

        for filename in files:
            pcd = o3d.io.read_point_cloud(filename)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            points = np.array(pcd.points)
            for i in range(0, len(points), retranche):
                print(f"point: {len(points[i : i+retranche])} filename: {filename}")
                self.anc_points['test'] += [points[i : i+retranche]]
                self.ids_list['test'] += [filename]
                self.num_test += 1

    def get_batch_gen(self, split, config):
        """
        Generates a data batch for training or testing.

        Args:
            split (str): Dataset split ('train' or 'test').
            config (dict): Configuration for the generator.

        Returns:
            tuple: (random_balanced_gen, gen_types, gen_shapes)
                - random_balanced_gen (generator): Yields batches of data.
                - gen_types (tuple): Data types for the generator output.
                - gen_shapes (tuple): Shapes for the generator output.
        """
        def random_balanced_gen():
            gen_indices = np.arange(self.num_test)
            for p_i in gen_indices:
                anc_id = self.ids_list['test'][p_i]
                pos_id = self.ids_list['test'][p_i]
                anc_points = self.anc_points['test'][p_i].astype(np.float32)
                pos_points = self.anc_points['test'][p_i].astype(np.float32)
                anc_keypts = np.array([])
                pos_keypts = np.array([])
                yield (np.concatenate([anc_points, pos_points], axis=0),
                       anc_keypts, pos_keypts,
                       np.array([p_i, p_i], dtype=np.int32),
                       np.array([anc_points.shape[0], pos_points.shape[0]]),
                       np.array([anc_id, pos_id]),
                       np.concatenate([anc_points, pos_points], axis=0))

        gen_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.float32)
        gen_shapes = ([None, 3], [None], [None], [None], [None], [None], [None, 3])
        return random_balanced_gen, gen_types, gen_shapes

    def get_tf_mapping(self, config):
        """
        Returns a TensorFlow mapping function to transform raw data into model inputs.

        Args:
            config (dict): Configuration for the mapping.

        Returns:
            function: A tf_map function that processes raw data into formatted inputs.
        """
        def tf_map(anc_points, anc_keypts, pos_keypts, obj_inds, stack_lengths, ply_id, backup_points):
            batch_inds = self.tf_get_batch_inds(stack_lengths)
            stacked_features = tf.ones((tf.shape(anc_points)[0], 1), dtype=tf.float32)
            anchor_input_list = self.tf_descriptor_input(config, anc_points, stacked_features, stack_lengths, batch_inds)
            return anchor_input_list + [stack_lengths, anc_keypts, pos_keypts, ply_id, backup_points]

        return tf_map

config = Config()

d3feat_model_path = "/lustre/fswork/projects/rech/dki/ujo91el/SpectralGV_D3Feat/models/D3Feat"
config.load(d3feat_model_path)

retranche = 10000000
name_point_cloud_files = ['/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/00/ply/000000.ply']
dataset = MiniDataset(files=name_point_cloud_files, retranche=retranche, voxel_size=0.1)
dataset.init_test_input_pipeline(config)

model = KernelPointFCNN(dataset.flat_inputs, config)


snap_path = os.path.join(d3feat_model_path, 'snapshots')
snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f.endswith('.meta')]
chosen_step = np.sort(snap_steps)[-1]

chosen_snap = os.path.join(d3feat_model_path, 'snapshots', f'snap-{chosen_step}')
import pdb; pdb.set_trace()

class RegTester:
    """
    A class to test and generate descriptors for point cloud models.
    """

    def __init__(self, model, restore_snap=None):
        """
        Initializes the RegTester with a TensorFlow session and optionally restores a model snapshot.

        Args:
            model: The model to be tested.
            restore_snap (str, optional): Path to a model snapshot to restore. If None, no snapshot is restored.

        Attributes:
            saver (tf.train.Saver): Saver object for TensorFlow variables.
            sess (tf.Session): TensorFlow session for running operations.
        """
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        cProto = tf.ConfigProto(
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=2,
            device_count={'CPU': 4}
        )
        self.sess = tf.Session(config=cProto)
        self.sess.run(tf.global_variables_initializer())
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

    def generate_descriptor(self, model, dataset, folder_rotation):
        """
        Generates descriptors for each point cloud in the dataset and saves them to .npz files.

        Args:
            model: The model used to generate descriptors.
            dataset: The dataset containing point clouds to process.
            folder_rotation (str): Path to the folder where results will be saved.

        Returns:
            None: Results are saved directly to .npz files in the specified folder.
        """
        self.sess.run(dataset.test_init_op)
        results = []
        for i in range(dataset.num_test):
            ops = [model.anchor_inputs, model.out_features, model.out_scores, model.anc_id]
            [inputs, features, scores, anc_id] = self.sess.run(ops, {model.dropout_prob: 1.0})
            scores_first_pcd = scores[inputs['in_batches'][0][:-1]]
            selected_keypoints_id = np.argsort(scores_first_pcd, axis=0)[:].squeeze()
            keypts_score = scores[selected_keypoints_id]
            keypts_loc = inputs['backup_points'][selected_keypoints_id]
            anc_features = features[selected_keypoints_id]
            base_filename_name = os.path.splitext(os.path.basename(anc_id.decode("utf-8")))[0]

            base_filename = os.path.join(folder_rotation, base_filename_name)
            if not os.path.exists(os.path.dirname(base_filename)):
                os.makedirs(os.path.dirname(base_filename))
            self.save_or_append_npz(folder_rotation, base_filename_name, keypts_loc, anc_features, keypts_score)

    def save_or_append_npz(self, folder_rotation, base_filename_name, keypts_loc, anc_features, keypts_score):
        """
        Saves or appends keypoints, features, and scores to a .npz file.

        Args:
            folder_rotation (str): Path to the folder where the file will be saved.
            base_filename_name (str): Base name of the file (without extension).
            keypts_loc (numpy.ndarray): Array of keypoint locations.
            anc_features (numpy.ndarray): Array of features for each keypoint.
            keypts_score (numpy.ndarray): Array of scores for each keypoint.

        Returns:
            None: Data is saved to a .npz file.
        """
        base_filename = os.path.join(folder_rotation, base_filename_name)
        os.makedirs(os.path.dirname(base_filename), exist_ok=True)
        filepath = base_filename + ".npz"
        if os.path.exists(filepath):
            # Load existing data
            with np.load(filepath) as data:
                old_keypts = data['keypts']
                old_features = data['features']
                old_scores = data['scores']
            # Concatenate with new data
            new_keypts = np.concatenate((old_keypts, keypts_loc), axis=0)
            new_features = np.concatenate((old_features, anc_features), axis=0)
            new_scores = np.concatenate((old_scores, keypts_score), axis=0)
        else:
            # No existing file, use new data
            new_keypts = keypts_loc
            new_features = anc_features
            new_scores = keypts_score
        # Save to file (overwrite or create new)
        np.savez_compressed(
            filepath,
            keypts=new_keypts,
            features=new_features,
            scores=new_scores,
        )

import pdb; pdb.set_trace()
tester = RegTester(model, restore_snap=chosen_snap)

import pdb; pdb.set_trace()
folder_rotation = ""
tester.generate_descriptor(model, dataset, )
"""
tester.generate_descriptor(model, dataset, self.folder_rotation)
"""