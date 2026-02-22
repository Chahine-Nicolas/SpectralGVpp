# Basic libs
import os
import tensorflow as tf
import numpy as np
import time
import glob
import random
import pickle
import copy
import open3d
import json

# Dataset parent class
from datasets.common import Dataset
from datasets.ThreeDMatch import rotate
import pandas as pd


kitti_icp_cache = {}
kitti_cache = {}


def make_open3d_point_cloud(xyz, color=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = open3d.utility.Vector3dVector(color)
    return pcd


def make_open3d_feature(data, dim, npts):
    feature = open3d.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.astype('d').transpose()
    return feature


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = open3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


class LidarDataset(Dataset):
    AUGMENT = None
    DATA_FILES = {
        'train': 'data/lidar/config/train_lidar.txt',
        'val': 'data/lidar/config/val_lidar.txt',
        'test': 'data/lidar/config/test_lidar.txt'
    }
    TEST_RANDOM_ROTATION = True
    IS_ODOMETRY = True
    MAX_TIME_DIFF = 3

    def __init__(self,query, map,zone, input_threads=8, first_subsampling_dl=0.3, load_test=False):
        Dataset.__init__(self, 'LIDAR')
        self.data_util = 'LIDAR'
        self.network_model = 'descriptor'
        self.num_threads = input_threads
        self.load_test = load_test
        self.root = '/home/kdeneuville/Documents/SpectralGV/nuage_lidar/lidar_zone'
        self.icp_path = 'data/lidar/icp'
        self.voxel_size = first_subsampling_dl
        self.matching_search_voxel_size = first_subsampling_dl * 1.5
        

        # Initiate containers
        self.anc_points = {'train': [], 'val': [], 'test': []}
        self.files = {'train': [], 'val': [], 'test': []}

        if self.load_test:
            self.prepare_lidar_ply(query, map,zone,'test')
        else:
            self.prepare_lidar_ply(query, map,zone, split='train')
            self.prepare_lidar_ply(query, map ,zone,split='val')

    def prepare_lidar_ply(self,query, map,zone, split='train'):
        max_time_diff = self.MAX_TIME_DIFF
        subset_names = open(self.DATA_FILES[split]).read().split()
        
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.root +  f'/sequences/{drive_id}/bin/*.bin')
            
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = [(os.path.split(fname)[-1][:-4]) for fname in fnames]
            self.create_poses_file(inames, 20, drive_id)
                      

            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            """
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))
            more_than_10 = pdist > 10
            curr_time = inames[0]
            
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files[split].append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1
        # for dirname in subset_names:
        #     drive_id = int(dirname)
        #     inames = self.get_all_scan_ids(drive_id)
        #     for start_time in inames:
        #         for time_diff in range(2, max_time_diff):
        #             pair_time = time_diff + start_time
        #             if pair_time in inames:
        #                 self.files[split].append((drive_id, start_time, pair_time))

        """
        chemin_vers_fichier_json = "/home/kdeneuville//Documents/SpectralGV/nuage_lidar/lidar_zone/dictionaire/dictionaire_5.json"
        with open(chemin_vers_fichier_json, 'r', encoding='utf-8') as fichier:
            dictionnaire = json.load(fichier)

        
        start_time = dictionnaire[query]
        
        for i in range (len(map)):
            pair_time = dictionnaire[map[i]]
            self.files[split].append((zone, start_time, pair_time,query, map[i]))
        

        
        
        
        if split == 'train':
            self.num_train = len(self.files[split])
            print("Num_train", self.num_train)
        elif split == 'val':
            self.num_val = len(self.files[split])
            print("Num_val", self.num_val)
        else:
            self.num_test = len(self.files[split])
            print("Num_test", self.num_test)

        for idx in range(len(self.files[split])):
            drive = self.files[split][idx][0]
            
            filename = self._get_velodyne_fn(drive, self.files[split][idx][3])
            xyzr = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
            xyz = xyzr[:, :3]
            
            self.anc_points[split] += [xyz]

    def get_batch_gen(self, split, config):

        
        def random_balanced_gen():
            
            # Initiate concatenation lists
            anc_points_list = []
            pos_points_list = []
            anc_keypts_list = []
            pos_keypts_list = []
            backup_anc_points_list = []
            backup_pos_points_list = []
            ti_list = []
            ti_list_pos = []
            batch_n = 0
            

            # Initiate parameters depending on the chosen split
            if split == 'train':
                gen_indices = np.random.permutation(self.num_train)
                
                # gen_indices = np.arange(self.num_train)

            elif split == 'val':
                gen_indices = np.random.permutation(self.num_val)
                # gen_indices = np.arange(self.num_val)

            elif split == 'test':
                # gen_indices = np.random.permutation(self.num_test)
                gen_indices = np.arange(self.num_test)
                


            print(gen_indices)
            # Generator loop
            
            for p_i in gen_indices:

                if split == 'test':
                    aligned_anc_points, aligned_pos_points, anc_points, pos_points, matches, trans, flag = self.__getitem__(split, p_i)
                    
                    if flag == False:
                        continue
                else:
                    aligned_anc_points, aligned_pos_points, anc_points, pos_points, matches, trans, flag = self.__getitem__(split, p_i)
                    if flag == False:
                        continue

                anc_id = str(self.files[split][p_i][0]) + "@" + str(self.files[split][p_i][3])
                
                pos_id = str(self.files[split][p_i][0]) + "@" + str(self.files[split][p_i][4])
                # the backup_points shoule be in the same coordinate
                backup_anc_points = aligned_anc_points
                backup_pos_points = aligned_pos_points
                if split == 'test':
                    anc_keypts = np.array([])
                    pos_keypts = np.array([])
                    
                else:
                    # input to the network should be in different coordinates
                    anc_keypts = matches[:, 0]
                    pos_keypts = matches[:, 1]
                    selected_ind = np.random.choice(range(len(anc_keypts)), config.keypts_num, replace=False)
                    anc_keypts = anc_keypts[selected_ind]
                    pos_keypts = pos_keypts[selected_ind]
                    pos_keypts += len(anc_points)

                if split == 'train' or split == 'val':
                    # data augmentations: noise
                    anc_noise = np.random.rand(anc_points.shape[0], 3) * config.augment_noise
                    pos_noise = np.random.rand(pos_points.shape[0], 3) * config.augment_noise
                    anc_points += anc_noise
                    pos_points += pos_noise
                    # data augmentations: rotation
                    anc_points = rotate(anc_points, num_axis=config.augment_rotation)
                    pos_points = rotate(pos_points, num_axis=config.augment_rotation)
                    # data augmentations: scale
                    scale = config.augment_scale_min + (config.augment_scale_max - config.augment_scale_min) * random.random()
                    anc_points = scale * anc_points
                    pos_points = scale * pos_points
                    # data augmentations: translation
                    anc_points = anc_points + np.random.uniform(-config.augment_shift_range, config.augment_shift_range, 3)
                    pos_points = pos_points + np.random.uniform(-config.augment_shift_range, config.augment_shift_range, 3)

                # Add data to current batch
                print("#########################################################")
                anc_points_list += [anc_points]

                anc_keypts_list += [anc_keypts]

                pos_points_list += [pos_points]

                pos_keypts_list += [pos_keypts]

                backup_anc_points_list += [backup_anc_points]
                
                backup_pos_points_list += [backup_pos_points]
                
          
                ti_list += [p_i]

                ti_list_pos += [p_i]
             
                

                yield (np.concatenate(anc_points_list + pos_points_list, axis=0),  # anc_points
                       np.concatenate(anc_keypts_list, axis=0),  # anc_keypts
                       np.concatenate(pos_keypts_list, axis=0),
                       np.array(ti_list + ti_list_pos, dtype=np.int32),  # anc_obj_index
                       np.array([tp.shape[0] for tp in anc_points_list] + [tp.shape[0] for tp in pos_points_list]),  # anc_stack_length 
                       np.array([anc_id, pos_id]),
                       np.concatenate(backup_anc_points_list + backup_pos_points_list, axis=0),
                       np.array(trans)
                       )
                # print("\t Yield ", anc_id, pos_id)
                anc_points_list = []
                pos_points_list = []
                anc_keypts_list = []
                pos_keypts_list = []
                backup_anc_points_list = []
                backup_pos_points_list = []
                ti_list = []
                ti_list_pos = []
                import time
                # time.sleep(0.3)

        # Generator types and shapes
        gen_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.float32, tf.float32)
        gen_shapes = ([None, 3], [None], [None], [None], [None], [None], [None, 3], [4, 4])
        
        """
        with open("output_0_3.txt", "w") as f:
            for data in random_balanced_gen():
                print("anc_points:", data[0])
                print("anc_keypts:", data[1])
                print("pos_keypts:", data[2])
                print("anc_obj_index:", data[3])
                print("anc_stack_length:", data[4])
                print("anc_id and pos_id:", data[5])
                print("backup_points:", data[6])
                print("trans:", data[7])
                f.write("anc_points: {}\n".format(data[0]))
                f.write("anc_keypts: {}\n".format(data[1]))
                f.write("pos_keypts: {}\n".format(data[2]))
                f.write("anc_obj_index: {}\n".format(data[3]))
                f.write("anc_stack_length: {}\n".format(data[4]))
                f.write("anc_id and pos_id: {}\n".format(data[5]))
                f.write("backup_points: {}\n".format(data[6]))
                f.write("trans: {}\n".format(data[7]))
                f.write("\n")
        
        """

        return random_balanced_gen, gen_types, gen_shapes

    
    def create_poses_file(self , list_of_strings, decoupage, zone):
        """
        Creates a poses.txt file in the poses directory with the calculated pose matrices.

        :param list_of_strings: List of lists of strings containing the data to calculate the poses.
        :param decoupage: Value of decoupage used for the calculation of positions.
        """
        dictionary_id = {}
        ndx = 0

        # Create the poses directory if it does not exist
        poses_dir = os.path.join(self.root, 'poses')
        if not os.path.exists(poses_dir):
            os.makedirs(poses_dir)

        dictionaire_dir = os.path.join(self.root, 'dictionaire')
        if not os.path.exists(dictionaire_dir):
            os.makedirs(dictionaire_dir)

        # Open the file in write mode
        file_path = os.path.join(poses_dir, f'poses_{zone}.txt')
        with open(file_path, 'w') as file:
            for e in list_of_strings:
                string_list = e.split("_")
                dictionary_id[e] = ndx

                # Calculate pose_x and pose_y
                pose_x = ((int(string_list[2]) - 656) * 1000 + int(string_list[8]) * decoupage + decoupage / 2)
                pose_y = ((int(string_list[3]) - 6860) * 1000 + int(string_list[9]) * decoupage + decoupage / 2)

                # Create the pose matrix
                pose = np.array([
                    [1, 0, 0, pose_x],
                    [0, 1, 0, pose_y],
                    [0, 0, 1, 0]
                ])
                ndx +=1

                # Write the pose matrix to the file
                np.savetxt(file, pose, fmt='%.8e', newline=' ')
                file.write('\n')

        json_file_path = os.path.join(dictionaire_dir, f'dictionaire_{zone}.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(dictionary_id, json_file, indent=4)

        print("poses.txt file created successfully in the poses directory.")

    
    
    def get_tf_mapping(self, config):
        def tf_map(anc_points, anc_keypts, pos_keypts, obj_inds, stack_lengths, ply_id, backup_points, trans):
            batch_inds = self.tf_get_batch_inds(stack_lengths)
            stacked_features = tf.ones((tf.shape(anc_points)[0], 1), dtype=tf.float32)
            anchor_input_list = self.tf_descriptor_input(config,
                                                         anc_points,
                                                         stacked_features,
                                                         stack_lengths,
                                                         batch_inds)
            return anchor_input_list + [stack_lengths, anc_keypts, pos_keypts, ply_id, backup_points, trans]
        


        return tf_map

    def get_all_scan_ids(self, drive_id):
        if self.IS_ODOMETRY:
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        else:
            fnames = glob.glob(self.root + '/' + self.date +
                               '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    def __getitem__(self, split, idx):
        
        drive = self.files[split][idx][0]
        t0, t1 = self.files[split][idx][1], self.files[split][idx][2]
        t0_name, t1_name = self.files[split][idx][3], self.files[split][idx][4]
        
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        #positions[0] = positions[0]+np.array([[0,0,0,all_odometry[0][3]],[0,0,0,all_odometry[0][7]],[0,0,0,0],[0,0,0,0]])
        #positions[1] = positions[1]+np.array([[0,0,0,all_odometry[1][3]],[0,0,0,all_odometry[1][7]],[0,0,0,0],[0,0,0,0]])
        fname0 = self._get_velodyne_fn(drive, t0_name )
        fname1 = self._get_velodyne_fn(drive, t1_name )
        

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
        # Fonction pour réduire une valeur à un multiple de 3
        def reduce_to_multiple_of_three(value):
            remainder = value % 3
            print(remainder)
            if remainder == 0:
                return value
            else:
                return value - remainder

        
        xyz0 = xyzr0[:, :3]- np.array([reduce_to_multiple_of_three(np.mean(xyzr0[:,0])) ,reduce_to_multiple_of_three(np.mean(xyzr0[:,1])),reduce_to_multiple_of_three(np.mean(xyzr0[:,2]))])#reduce_to_multiple_of_three(all_odometry[0][7]) ,0])#- np.array([0.3,0,0])
        xyz1 = xyzr1[:, :3]- np.array([reduce_to_multiple_of_three(np.mean(xyzr1[:,0])),reduce_to_multiple_of_three(np.mean(xyzr1[:,1])),reduce_to_multiple_of_three(np.mean(xyzr1[:, 2]))])#reduce_to_multiple_of_three(all_odometry[1][3]) reduce_to_multiple_of_three(all_odometry[1][7]) ,0])#- np.array([0.3,0,0])
        
        xyz0[:, 0] = xyz0[:, 0]/(np.max(xyz0[:, 0]) - np.min(xyz0[:, 0]))
        xyz0[:, 1] = xyz0[:, 1]/(np.max(xyz0[:, 1]) - np.min(xyz0[:, 1])) 
        xyz0[:, 2] = xyz0[:, 2]/(np.max(xyz0[:, 2]) - np.min(xyz0[:, 2]))
        xyz1[:, 0] = xyz1[:, 0]/(np.max(xyz1[:, 0]) - np.min(xyz1[:, 0]))
        xyz1[:, 1] = xyz1[:, 1]/(np.max(xyz1[:, 1]) - np.min(xyz1[:, 1]))
        xyz1[:, 2] = xyz1[:, 2]/(np.max(xyz1[:, 2]) - np.min(xyz1[:, 2]))
        
    

        
        
        #print(np.array([reduce_to_multiple_of_three(all_odometry[0][3]) ,reduce_to_multiple_of_three(all_odometry[0][7]) ,0]),np.array([reduce_to_multiple_of_three(all_odometry[1][3]) ,reduce_to_multiple_of_three(all_odometry[1][7]) ,0]))
        """
        phi_pas = 0
        theta_pas = 0
        gamma = 0
        rotation_center_xyz0 = np.array([np.mean(xyz0[:, 0]), np.mean(xyz0[:, 1]), np.mean(xyz0[:, 2])])
        rotation_center_xyz1 = np.array([np.mean(xyz1[:, 0]), np.mean(xyz1[:, 1]), np.mean(xyz1[:, 2])])


        def rotation_matrix_x(phi):
            return np.array([
                [1, 0, 0],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi), np.cos(phi)]
            ])

        def rotation_matrix_y(theta):
            return np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])

        def rotation_matrix_z(gamma):
            return np.array([
                [np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1]
            ])

        def rotate_points(points, phi, theta, gamma, rotation_center):
            phi = np.radians(phi)
            theta = np.radians(theta)
            gamma = np.radians(gamma)

            R_x = rotation_matrix_x(phi)
            R_y = rotation_matrix_y(theta)
            R_z = rotation_matrix_z(gamma)
            R = R_z @ R_y @ R_x

            coordinates = points[:, :3]

            # Centrage sur le centre de rotation extrait du nom
            centered_coordinates = coordinates - rotation_center
            rotated_coordinates = (R @ centered_coordinates.T).T + rotation_center

            rotated_points = rotated_coordinates

            return rotated_points


        xyz0 = rotate_points(xyz0, phi_pas, theta_pas, gamma, rotation_center_xyz0)
        xyz1 = rotate_points(xyz1, phi_pas, theta_pas, gamma, rotation_center_xyz1)

        # Sauvegarder les informations nécessaires pour la rotation inverse dans un fichier JSON
        rotation_info = {
            'phi_pas': phi_pas,
            'theta_pas': theta_pas,
            'gamma': gamma,
            'rotation_center_xyz0': rotation_center_xyz0.tolist(),
            'rotation_center_xyz1': rotation_center_xyz1.tolist()
        }

        with open('rotation_info.json', 'w') as f:
            json.dump(rotation_info, f)
        
        """
        key = f'{drive}_{t0_name}_{t1_name}' 
        
        if not os.path.exists(self.icp_path):
            # Créer le dossier
            os.makedirs(self.icp_path)
    
        filename = self.icp_path + '/' + key + '.npy'
        
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):
                if self.IS_ODOMETRY:
                    M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                         @ np.linalg.inv(self.velo2cam)).T

                xyz0_t = self.apply_transform(xyz0, M)
                pcd0 = make_open3d_point_cloud(xyz0_t)
                pcd1 = make_open3d_point_cloud(xyz1)
                reg = open3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                           open3d.registration.TransformationEstimationPointToPoint(),
                                                           open3d.registration.ICPConvergenceCriteria(max_iteration=200))
                pcd0.transform(reg.transformation)
                # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
                M2 = M @ reg.transformation
                
                # open3d.draw_geometries([pcd0, pcd1])
                # write to a file

                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:

            M2 = kitti_icp_cache[key]


        trans = M2

        
        

        pcd0 = make_open3d_point_cloud(xyz0)
        pcd1 = make_open3d_point_cloud(xyz1)
    
        pcd0 = open3d.voxel_down_sample(pcd0, self.voxel_size)
        
        pcd1 = open3d.voxel_down_sample(pcd1, self.voxel_size)
        unaligned_anc_points = np.array(pcd0.points)#+ np.array([all_odometry[0][3] *2 ,all_odometry[0][7]*2 ,0])
        unaligned_pos_points = np.array(pcd1.points)#+ np.array([all_odometry[1][3] *2 ,all_odometry[1][7]*2 ,0]) 

        

        # Get matches
        # if True:
        if split == 'train' or split == 'val':
            matching_search_voxel_size = self.matching_search_voxel_size
            matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
            if len(matches) < 1024:
                # raise ValueError(f"{drive}, {t0}, {t1}, {len(matches)}/{len(pcd0.points)}")
                print(f"Not enought corr: {drive}, {t0}, {t1}, {len(matches)}/{len(pcd0.points)}")
                return (None, None, None, None, None, None, False)
        else:
            matches = np.array([])

        # align the two point cloud into one corredinate system.
        matches = np.array(matches)
        #pcd0.transform(trans)
        anc_points = np.array(pcd0.points)#+ np.array([all_odometry[0][3] *2 ,all_odometry[0][7]*2 ,0])
        pos_points = np.array(pcd1.points)#+ np.array([all_odometry[0][3] *2 ,all_odometry[0][7]*2 ,0]) 
        
        
        return (anc_points, pos_points, unaligned_anc_points, unaligned_pos_points, matches, trans, True)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T-[1000]
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                    1, 0, 0, 0, 1,
                    0, 0, 0, 1
                ]).reshape(3, 3)
            T = np.array([0, 0, 0]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0,0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        data_path = self.root + f'/poses/poses_{drive}.txt' 
        
        if data_path not in kitti_cache:
            kitti_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return kitti_cache[data_path]
        else:
            return kitti_cache[data_path][indices]
        

    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)            
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))

        return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        fname = self.root + f'/sequences/{drive}/bin/{t}.bin'
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)
