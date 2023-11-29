'''
Author: Zexi Liu
Date: 2023-11-28 17:11:02
LastEditors: Zexi Liu
LastEditTime: 2023-11-29 18:32:36
FilePath: /seg_data_tools/main.py
Description:

Copyright (c) 2023 by Uisee, All Rights Reserved.
'''

import os
import numpy as np
from tqdm import tqdm

INPUT_BIN_FOLDER = '/data/code/seg_data_tools/data/seg_pts'
INPUT_LABEL_FOLDER = '/data/code/seg_data_tools/data/seg_results'
INPUT_LIDAR_STATE = '/data/code/seg_data_tools/data/ml_lidar_state'


def point_local2global(local_x, local_y, pose_x, pose_y, theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    global_x = pose_x + sin_theta * local_x + cos_theta * local_y
    global_y = pose_y + sin_theta * local_y - cos_theta * local_x
    return [global_x, global_y]

def main():
    input_bin_folder = os.listdir(INPUT_BIN_FOLDER)
    input_label_folder = os.listdir(INPUT_LABEL_FOLDER)
    assert(len(input_bin_folder) == len(input_label_folder))
    bin_files = np.sort(input_bin_folder)
    label_files = np.sort(input_label_folder)
    lidar_state = np.loadtxt(INPUT_LIDAR_STATE, dtype=np.str)
    assert(len(lidar_state) == len(input_label_folder))

    for i in tqdm(range(len(bin_files))):
        bin_file_path = os.path.join(INPUT_BIN_FOLDER, bin_files[i])
        label_file_path = os.path.join(INPUT_LABEL_FOLDER, label_files[i])
        bin_file = np.fromfile(bin_file_path)
        label_file = np.fromfile(label_file_path)

        a = 1


if __name__ == '__main__':
    main()
    a = 1