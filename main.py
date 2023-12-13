'''
Author: Zexi Liu
Date: 2023-11-28 17:11:02
LastEditors: Zexi Liu
LastEditTime: 2023-12-12 19:04:18
FilePath: /seg_data_tools/main.py
Description:

Copyright (c) 2023 by Uisee, All Rights Reserved.
'''

import os
import numpy as np
from tqdm import tqdm
import pypcd

INPUT_BIN_FOLDER = '/data/code/seg_data_tools/data/seg_pts'
# INPUT_BIN_FOLDER = '/data/code/seg_data_tools/data/lidar_20231113111338'
INPUT_LABEL_FOLDER = '/data/code/seg_data_tools/data/seg_results'
INPUT_LIDAR_STATE = '/data/code/seg_data_tools/data/ml_lidar_state'
USE_HISTORY_FRAME = 5

def point_local2global(pt_local, veh_pose):
    veh_pose_x, veh_pose_y, veh_pose_theta = float(veh_pose[0]), float(veh_pose[1]), float(veh_pose[2])
    sin_theta = np.sin(veh_pose_theta)
    cos_theta = np.cos(veh_pose_theta)
    pt_global_x = veh_pose_x + sin_theta * pt_local[0] + cos_theta * pt_local[1]
    pt_global_y = veh_pose_y + sin_theta * pt_local[1] - cos_theta * pt_local[0]
    pt_global_z = pt_local[2]
    return [pt_global_x, pt_global_y, pt_global_z]

def point_global2local(pt_global, veh_pose):
    veh_pose_x, veh_pose_y, veh_pose_theta = float(veh_pose[0]), float(veh_pose[1]), float(veh_pose[2])
    dx = pt_global[0] - veh_pose_x
    dy = pt_global[1] - veh_pose_y
    sin_theta = np.sin(veh_pose_theta)
    cos_theta = np.cos(veh_pose_theta)
    pt_local_x = sin_theta * dx - cos_theta * dy
    pt_local_y = cos_theta * dx + sin_theta * dy
    pt_local_z = pt_global[2]
    return [pt_local_x, pt_local_y, pt_local_z]

def cal_relative_pose(pt, cur_pose, start_pose):
    x = pt[0]
    y = pt[1]
    dx = cur_pose[0] - start_pose[0]
    dy = cur_pose[1] - start_pose[1]
    sin_theta_cur = np.sin(cur_pose[2])
    cos_theta_cur = np.cos(cur_pose[2])
    sin_theta_start = np.sin(start_pose[2])
    cos_theta_start = np.cos(start_pose[2])
    relative_x = sin_theta_start*(dx + sin_theta_cur*x + cos_theta_cur*y) - cos_theta_start*(dy + sin_theta_cur*y - cos_theta_cur*x)
    relative_y = cos_theta_start*(dx + sin_theta_cur*x + cos_theta_cur*y) + sin_theta_start*(dy + sin_theta_cur*y - cos_theta_cur*x)
    return [relative_x, relative_y, pt[2]]

def read_lidar_state(lidar_state):
    gps_pos = lidar_state[:, 5:8]
    navi_pos = lidar_state[:, 8:11]
    # only use state 4 (RTK high precision)
    gps_state = lidar_state[:, 11]
    # only use state 3 (navi high precision)
    position_state = lidar_state[:, 12]

    return gps_pos, navi_pos, gps_state, position_state


def main():
    input_bin_folder = os.listdir(INPUT_BIN_FOLDER)
    input_label_folder = os.listdir(INPUT_LABEL_FOLDER)
    # assert(len(input_bin_folder) == len(input_label_folder))
    bin_files = np.sort(input_bin_folder)
    label_files = np.sort(input_label_folder)
    lidar_state = np.loadtxt(INPUT_LIDAR_STATE, dtype=np.str)
    # assert(len(lidar_state) == len(input_label_folder))

    gps_pos, navi_pos, gps_state, position_state = read_lidar_state(lidar_state)
    start_veh_pose = [float(navi_pos[0][0]), float(navi_pos[0][1]), float(navi_pos[0][2])]
    all_pt = []
    for i in tqdm(range(len(bin_files))):
        if gps_state[i] != '4' and position_state[i] != '3':
            continue
        elif gps_state[i] == '4':
            veh_pose = [float(navi_pos[i][0]), float(navi_pos[i][1]), float(navi_pos[i][2])]
        elif position_state[i] == '3':
            veh_pose = [float(position_state[i][0]), float(position_state[i][1]), float(position_state[i][2])]

        bin_file_path = os.path.join(INPUT_BIN_FOLDER, bin_files[i])
        label_file_path = os.path.join(INPUT_LABEL_FOLDER, label_files[i])

        # pcd = pypcd.PointCloud.from_path(bin_file_path)
        # bin_file = pypcd.get_points_xyz(pcd.pc_data)

        bin_file = np.fromfile(bin_file_path, dtype=np.float32)
        bin_file = bin_file.reshape(-1,8)[:, :4]


        label_file = np.fromfile(label_file_path, dtype=np.int32)
        for j in range(len(bin_file)):
            # pt_global_pos = point_local2global(bin_file[j], veh_pose)
            # pt_relative_pos = point_global2local(pt_global_pos, start_veh_pose)
            pt_relative_pos = cal_relative_pose(bin_file[j], veh_pose, start_veh_pose)

            pt_relative_pos.append(label_file[j])
            all_pt.append(pt_relative_pos)
        save_all_pt = np.array(all_pt, dtype=np.float32)
        save_name = os.path.join('/data/code/seg_data_tools/test/', str(i).zfill(6) + '.bin')
        save_all_pt.tofile(save_name)
        a = 1

if __name__ == '__main__':
    main()
    a = 1