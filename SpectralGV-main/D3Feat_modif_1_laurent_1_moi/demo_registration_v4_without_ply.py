import glob
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
import os.path

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
        
        def voxel_downsample(points, voxel_size):
            coords = np.floor(points / voxel_size).astype(np.int32)
            _, idx = np.unique(coords, axis=0, return_index=True)
            return points[idx]

        for filename in files:
            #pcd = o3d.io.read_point_cloud(filename)
            #data = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
            #points = data[:, :3]   # ignore intensity


            data = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
            points = data[:, :3]
            points = voxel_downsample(points, voxel_size)
            
            #pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            #points = np.array(pcd.points)
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
            print("saved to ", folder_rotation, base_filename_name)
            

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




def save_keypoints_as_cloud(keypts, scores, filename_prefix, vox_size):
    """
    Saves keypoints as a colored point cloud (.ply) based on their scores.

    Args:
        keypts (numpy.ndarray): Array of keypoint coordinates, shape (N, 3).
        scores (numpy.ndarray): Array of scores for each keypoint, shape (N,).
        filename_prefix (str): Prefix for the output filename.
        vox_size (float): Voxel size used for downsampling, included in the output filename.

    Returns:
        None: Saves the colored point cloud to a .ply file.
    """
    top_k_indices = np.argsort(scores, axis=0)[:]
    top_keypts = keypts[top_k_indices].reshape(-1, 3)
    top_scores = scores[top_k_indices].reshape(-1)
    norm_scores = (top_scores - np.min(top_scores)) / (np.max(top_scores) - np.min(top_scores) + 1e-8)
    colormap = plt.get_cmap('coolwarm')
    colors = colormap(norm_scores)[:, :3]
    kp_cloud = o3d.io.geometry.PointCloud()
    kp_cloud.points = o3d.io.utility.Vector3dVector(top_keypts)
    kp_cloud.colors = o3d.io.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud(f"{filename_prefix}_top300_keypts_vxs{vox_size}.ply", kp_cloud)

class D3Feat:
    """
    A class to run D3Feat feature extraction on point cloud datasets.
    """

    def __init__(self, model_path, dataset_root, eval_set_query, eval_set_map, MEAN_SHIFT_p=False, voxel_size=0.3, reset=False, file_rotation='',debut=0):
        """
        Initializes the D3Feat feature extractor.

        Args:
            model_path (str): Path to the model configuration and snapshots.
            dataset_root (str): Root directory of the dataset.
            eval_set_query (list): List of query point cloud files to process.
            eval_set_map (list): List of map point cloud files to process.
            MEAN_SHIFT_p (bool, optional): Whether to use mean shift post-processing. Defaults to False.
            voxel_size (float, optional): Voxel size for downsampling. Defaults to 0.3.
            reset (bool, optional): If True, forces reprocessing of existing files. Defaults to False.
            file_rotation (str, optional): Subfolder name for saving results. Defaults to ''.

        Attributes:
            folder_rotation (str): Path to the directory where results will be saved.
        """
        self.env_path = os.getenv("WORK")
        self.model_path = os.path.join(self.env_path,model_path)
        self.dataset_root = dataset_root
        self.eval_set_query = eval_set_query
        self.eval_set_map = eval_set_map
        self.MEAN_SHIFT_p = MEAN_SHIFT_p
        self.voxel_size = voxel_size
        self.reset = reset
        self.debut = debut
        chemain_stokage_fichier = os.getenv("WORKSF")
        self.folder_rotation = os.path.join(chemain_stokage_fichier, f'descripteur_D3Feat/{file_rotation}')
        if self.reset and os.path.exists(self.folder_rotation):
            shutil.rmtree(self.folder_rotation)

    def run(self):
        """
        Runs the feature extraction pipeline for all point clouds in the dataset.

        Steps:
            1. Converts .bin files to .ply format if needed.
            2. Processes each point cloud to extract and save descriptors.

        Args:
            None: Uses instance attributes set in __init__.

        Returns:
            None: Results are saved to the folder specified in self.folder_rotation.
        """
        name_point_cloud_files = []
        config = Config()
        config.load(self.model_path)

        ply_directory = Path(self.eval_set_query[0].rel_scan_filepath).parents[1]
        ply_path = os.path.join(self.env_path, self.dataset_root, str(ply_directory) + "/ply")
        bin_path = os.path.join(self.env_path, self.dataset_root, str(ply_directory) + "/bin")

        
        
        # /lustre/fsn1/worksf/projects/rech/dki/ujo91el 
        print("ply_path ",ply_path)
        print("self.eval_set_query", len(self.eval_set_query))
        for ndx, e in tqdm(enumerate(self.eval_set_query)):
            if not os.path.exists(ply_path):
                os.makedirs(ply_path)
            last_element = os.path.basename(e.rel_scan_filepath)
            filename_without_extension = os.path.splitext(last_element)[0]
            ply_file_map = os.path.join(ply_path, filename_without_extension + ".ply")

            #if not os.path.exists(ply_file_map) or self.reset == True:
                #self.bin_to_ply_with_intensity(os.path.join(self.env_path, self.dataset_root, e.rel_scan_filepath), ply_file_map)
            
            name_point_cloud_files.append(os.path.join(self.env_path, bin_path, os.path.basename(e.rel_scan_filepath)[:-4] + ".bin"))
            #name_point_cloud_files.append(os.path.join(self.env_path, ply_path, os.path.basename(e.rel_scan_filepath)[:-4] + ".ply"))
   
            
        print("self.eval_set_map", len(self.eval_set_map))
        for ndx, e in tqdm(enumerate(self.eval_set_map)):
            last_element = os.path.basename(e.rel_scan_filepath)
            filename_without_extension = os.path.splitext(last_element)[0]
            ply_file_map = os.path.join(ply_path, filename_without_extension + ".ply")
            #if not os.path.exists(ply_file_map) or self.reset == True:
                
                #self.bin_to_ply_with_intensity(os.path.join(self.env_path, self.dataset_root, e.rel_scan_filepath), ply_file_map)
            name_point_cloud_files.append(os.path.join(self.env_path, bin_path, os.path.basename(e.rel_scan_filepath)[:-4] + ".bin"))
            #name_point_cloud_files.append(os.path.join(self.env_path, ply_path, os.path.basename(e.rel_scan_filepath)[:-4] + ".ply"))
   

        #retranche = 10000000
        retranche = 15000
        
        print("self.debut, len(name_point_cloud_files)", self.debut, len(name_point_cloud_files))

        
        import json
        path = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/splits/dsi_train_list_part2.json"
        #path ="/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/small_full_eval_list.json" # tile 5
        path ="/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/dsi_eval_list.json" # 
        path ="/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2/splits/dsi_eval_list_part4.json"
        
        with open(path, "r") as f:
            dataset_full = json.load(f)

        #self.folder_rotation = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/descripteur_D3Feat/rotation"
        self.folder_rotation = "/lustre/fsn1/projects/rech/dki/ujo91el/descripteur_D3Feat/lidarhd_v2_2m"
        
        for i in tqdm(range(len(dataset_full))):
            print(i, dataset_full[i:i+1])

            if os.path.isfile(self.folder_rotation + "/"+os.path.basename(dataset_full[i:i+1][0])[:-4] + ".npz"):
                continue
            
            dataset = MiniDataset(files=dataset_full[i:i+1], retranche=retranche, voxel_size=self.voxel_size)
            
            dataset.init_test_input_pipeline(config)
            
            model = KernelPointFCNN(dataset.flat_inputs, config)

            snap_path = os.path.join(self.model_path, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f.endswith('.meta')]
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(self.model_path, 'snapshots', f'snap-{chosen_step}')
            tester = RegTester(model, restore_snap=chosen_snap)
            
            tester.generate_descriptor(model, dataset, self.folder_rotation)
        
        """
        for i in tqdm(range(self.debut, len(name_point_cloud_files), 1)):
            dataset = MiniDataset(files=name_point_cloud_files[i:i+1], retranche=retranche, voxel_size=self.voxel_size)
            dataset.init_test_input_pipeline(config)
            model = KernelPointFCNN(dataset.flat_inputs, config)

            snap_path = os.path.join(self.model_path, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f.endswith('.meta')]
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(self.model_path, 'snapshots', f'snap-{chosen_step}')
            tester = RegTester(model, restore_snap=chosen_snap)
            tester.generate_descriptor(model, dataset, self.folder_rotation)
        """
    

    def bin_to_ply_with_intensity(self, bin_file, ply_file):
        """
        Converts a .bin file containing point cloud data to a .ply file.

        Args:
            bin_file (str): Path to the input .bin file.
            ply_file (str): Path to the output .ply file.

        Returns:
            None: Saves the point cloud data to a .ply file.
        """
        # Read data: assumes 4 floats per point (x, y, z, intensity)
        data = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

        header = f'''ply
format ascii 1.0
element vertex {len(data)}
property float x
property float y
property float z
property float intensity
end_header
'''
        print("ply_file ", ply_file)

        with open(ply_file, 'w') as f:
            f.write(header)
            for point in data:
                f.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")

class EvaluationTuple:
    """
    A tuple describing a single element in an evaluation set.
    """

    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array, pose: np.array = None):
        """
        Initializes an EvaluationTuple with timestamp, file path, position, and optional pose.

        Args:
            timestamp (int): Timestamp associated with the scan.
            rel_scan_filepath (str): Relative file path to the scan.
            position (np.array): 2D array representing x, y position in meters. Shape must be (2,).
            pose (np.array, optional): 6 DoF pose as a 4x4 pose matrix. Shape must be (4, 4). Defaults to None.

        Raises:
            AssertionError: If position or pose shapes are incorrect.
        """
        # position: x, y position in meters
        # pose: 6 DoF pose (as 4x4 pose matrix)
        assert position.shape == (2,)
        assert pose is None or pose.shape == (4, 4)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position
        self.pose = pose

    def to_tuple(self):
        """
        Converts the EvaluationTuple to a tuple.

        Returns:
            tuple: A tuple containing (timestamp, rel_scan_filepath, position, pose).
        """
        return self.timestamp, self.rel_scan_filepath, self.position, self.pose

class EvaluationSet:
    """
    A class representing an evaluation set, consisting of map and query elements.
    """

    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        """
        Initializes an EvaluationSet with optional query and map sets.

        Args:
            query_set (List[EvaluationTuple], optional): List of query EvaluationTuples. Defaults to None.
            map_set (List[EvaluationTuple], optional): List of map EvaluationTuples. Defaults to None.
        """
        self.query_set = query_set
        self.map_set = map_set

    def load(self, pickle_filepath: str):
        """
        Loads an evaluation set from a pickle file.

        Args:
            pickle_filepath (str): Path to the pickle file to load.

        Returns:
            None: Populates self.query_set and self.map_set with EvaluationTuples from the pickle file.
        """
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))
        self.map_set = []
        for e in map_l:
            self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))

def str2bool(v):
    """
    Converts a string to a boolean value.

    Args:
        v (str or bool): Input value to convert. Can be a boolean, or a string representing a boolean.

    Returns:
        bool: The converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the input string is not a recognized boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MinkLoc model')
    parser.add_argument('--dataset_root', type=str, required=False, default='', help='Path to the dataset root')
    parser.add_argument('--path_model_D3Feat', type=str, required=False, default='' )
    parser.add_argument('--dataset_type', type=str, required=False, default='kitti360', choices=['mulran', 'southbay', 'kitti', 'alita', 'kitti360','lidar', 'lidar_east', 'lidar_west'])
    parser.add_argument('--voxel_size', type=float, default=0.1, help='Voxel size for point cloud downsampling')
    parser.add_argument('--reset_fichier', type=str2bool, default=False, help='reset des fichier (True/False)')
    parser.add_argument('--MEAN_SHIFT_p', type=bool, default=False, help='Use mean shift for point cloud downsampling')
    parser.add_argument('--eval_set', type=str, required=False, default='', help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--file_rotation', type=str, required=False, default='rotation',help='Path to the folder containing rotation data.')
    parser.add_argument('--debut', type=int, required=False, default=0,help='Index of the first point cloud to process.')
    

    args = parser.parse_args()
    if args.eval_set =='':
        if args.dataset_type == 'kitti':
            args.eval_set = 'kitti_00_eval_tout_map.pickle'
        elif args.dataset_type == 'mulran':
            print(args.mulran_sequence)
            if args.mulran_sequence == 'sejong':
                args.eval_set = 'test_Sejong01_Sejong02.pickle'
            elif args.mulran_sequence == 'DCC' :
                args.eval_set = 'test_DCC_01_DCC_02_10.0_5.pickle'
        elif args.dataset_type == 'southbay':
            args.eval_set = 'test_SunnyvaleBigloop_1.0_5.pickle'
        elif args.dataset_type == 'alita':
            args.eval_set = 'test_val_5_0.01_5.pickle'
        elif args.dataset_type == 'kitti360':
            args.eval_set = 'kitti360_09_3.0_eval.pickle'
        elif args.dataset_type == 'lidar':
            args.eval_set = 'lidar_eval_test.pickle'
            #args.eval_set = 'test_pickle.pickle'
        #ajout
        elif args.dataset_type == 'lidar_east':
            args.eval_set = 'lidarhd_v2.pickle'
            eval_set_filepath = "/lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/SpectralGV-main/datasets/lidar/lidarhd_v2.pickle"
        elif args.dataset_type == 'lidar_west':
            args.eval_set = 'lidarhd_v3.pickle'
            eval_set_filepath = "/lustre/fswork/projects/rech/dki/ujo91el/code/SpectralGV_D3Feat/SpectralGV-main/datasets/lidar/lidarhd_v3.pickle"

    #eval_set_filepath = os.path.join(os.path.dirname(__file__), '../datasets/',args.dataset_type, args.eval_set)
    
    eval_set = EvaluationSet()
    eval_set.load(eval_set_filepath)


    d3feat = D3Feat(args.path_model_D3Feat, args.dataset_root ,eval_set.query_set , eval_set.map_set , args.MEAN_SHIFT_p, args.voxel_size , args.reset_fichier ,args.file_rotation , args.debut )
    d3feat.run()

