import open3d
import tensorflow as tf
import numpy as np
import os
import copy
import time
from utils.config import Config
from datasets.common import Dataset
from models.KPFCNN_model import KernelPointFCNN

open3d.set_verbosity_level(open3d.VerbosityLevel.Error)


class MiniDataset(Dataset):
    def __init__(self, files, voxel_size=0.03):
        Dataset.__init__(self, 'Mini')
        self.num_test = 0
        self.anc_points = {"train": [], "test": []}
        self.ids_list = {"train": [], "test": []}
        for filename in files:
            pcd = open3d.read_point_cloud(filename)
            pcd = open3d.voxel_down_sample(pcd, voxel_size=voxel_size)
            points = np.array(pcd.points)
            self.anc_points['test'] += [points]
            self.ids_list['test'] += [filename]
            self.num_test += 1

    def get_batch_gen(self, split, config):
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
                       anc_keypts,
                       pos_keypts,
                       np.array([p_i, p_i], dtype=np.int32),
                       np.array([anc_points.shape[0], pos_points.shape[0]]),
                       np.array([anc_id, pos_id]),
                       np.concatenate([anc_points, pos_points], axis=0))

        gen_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.float32)
        gen_shapes = ([None, 3], [None], [None], [None], [None], [None], [None, 3])

        return random_balanced_gen, gen_types, gen_shapes

    def get_tf_mapping(self, config):
        def tf_map(anc_points, anc_keypts, pos_keypts, obj_inds, stack_lengths, ply_id, backup_points):
            batch_inds = self.tf_get_batch_inds(stack_lengths)
            stacked_features = tf.ones((tf.shape(anc_points)[0], 1), dtype=tf.float32)
            anchor_input_list = self.tf_descriptor_input(config,
                                                         anc_points,
                                                         stacked_features,
                                                         stack_lengths,
                                                         batch_inds)
            return anchor_input_list + [stack_lengths, anc_keypts, pos_keypts, ply_id, backup_points]

        return tf_map


class RegTester:
    def __init__(self, model, restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        cProto = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=cProto)

        self.sess.run(tf.global_variables_initializer())

        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

    def generate_descriptor(self, model, dataset):
        self.sess.run(dataset.test_init_op)
        for i in range(dataset.num_test):
            ops = [model.anchor_inputs, model.out_features, model.out_scores, model.anc_id]
            [inputs, features, scores, anc_id] = self.sess.run(ops, {model.dropout_prob: 1.0})

            scores_first_pcd = scores[inputs['in_batches'][0][:-1]]
            selected_keypoints_id = np.argsort(scores_first_pcd, axis=0)[:].squeeze()
            keypts_score = scores[selected_keypoints_id]
            keypts_loc = inputs['backup_points'][selected_keypoints_id]
            anc_features = features[selected_keypoints_id]

            base_filename = os.path.splitext(os.path.basename(anc_id.decode("utf-8")))[0]
            np.savez_compressed(
                base_filename + ".npz",
                keypts=keypts_loc,
                features=anc_features,
                scores=keypts_score,
            )


def execute_global_registration(src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold):
    result = open3d.registration_ransac_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc,
        distance_threshold,
        open3d.TransformationEstimationPointToPoint(False), 4,
        [open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        open3d.RANSACConvergenceCriteria(4000000, 500))
    return result


def save_keypoints_as_cloud(keypts, scores, filename_pre
