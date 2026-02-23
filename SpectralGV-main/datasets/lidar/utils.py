# This file is directly copied from: https://github.com/jac99/Egonn/blob/main/datasets/kitti/utils.py

import numpy as np

def velo2cam():
    R = np.array([
        1, 0, 0, 0, 1,
        0, 0, 0, 1
    ]).reshape(3, 3)
    T = np.array([0, 0, 0]).reshape(3, 1)
    velo2cam = np.hstack([R, T])
    velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
    return velo2cam


def get_relative_pose(pose_1, pose_2):
    # as seen in https://github.com/chrischoy/FCGF
    M = (velo2cam() @ pose_1.T @ np.linalg.inv(pose_2.T) @ np.linalg.inv(velo2cam())).T
    return M
