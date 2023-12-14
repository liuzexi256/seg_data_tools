'''
Author: Zexi Liu
Date: 2023-11-28 17:11:02
LastEditors: Zexi Liu
LastEditTime: 2023-12-14 11:18:11
FilePath: /seg_data_tools/main.py
Description:

Copyright (c) 2023 by Uisee, All Rights Reserved.
'''

import os
import numpy as np
from tqdm import tqdm
import scipy.spatial as spt
import pypcd

INPUT_BIN_FOLDER = '/data/code/seg_data_tools/data/seg_pts'
# INPUT_BIN_FOLDER = '/data/code/seg_data_tools/data/lidar_20231113111338'
INPUT_LABEL_FOLDER = '/data/code/seg_data_tools/data/seg_results'
INPUT_BBOX_FOLDER = '/data/code/seg_data_tools/data/dl_res'
INPUT_LIDAR_STATE = '/data/code/seg_data_tools/data/ml_lidar_state'
USE_HISTORY_FRAME = 5

POINTS_CLASS_NAME = [
    'ignore', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'sidewalk',
    'terrain', 'other_flat', 'manmade', 'vegetation', 'pole', 'traffic sign', 'tricycle'
]

POINTS_BACKGROUND_NAME = [
    'ignore', 'barrier', 'traffic_cone', 'trailer', 'driveable_surface', 'sidewalk',
    'terrain', 'other_flat', 'manmade', 'vegetation', 'pole', 'traffic sign'
]

BBOX_CLASS_NAME = [
    'car', 'truck', 'bus', 'bicycle', 'triple_wheel', 'human', 'animal', 'traffic_cone', 'other'
]

LABEL_NAME_MAPPING = {
    'ignore': 'ignore',
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'bicycle': 'bicycle',
    'triple_wheel': 'tricycle',
    'human': 'pedestrian',
    'animal': 'ignore',
    'traffic_cone': 'traffic_cone',
    'other': 'ignore',
}

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

def cal_relative_pose(pts, labels, cur_pose, start_pose, id):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    ids = np.array([id]*len(pts))
    dx = cur_pose[0] - start_pose[0]
    dy = cur_pose[1] - start_pose[1]
    sin_theta_cur = np.sin(cur_pose[2])
    cos_theta_cur = np.cos(cur_pose[2])
    sin_theta_start = np.sin(start_pose[2])
    cos_theta_start = np.cos(start_pose[2])
    relative_x = sin_theta_start*(dx + sin_theta_cur*x + cos_theta_cur*y) - cos_theta_start*(dy + sin_theta_cur*y - cos_theta_cur*x)
    relative_y = cos_theta_start*(dx + sin_theta_cur*x + cos_theta_cur*y) + sin_theta_start*(dy + sin_theta_cur*y - cos_theta_cur*x)
    new_pts = np.hstack((relative_x.reshape(-1, 1), relative_y.reshape(-1, 1), z.reshape(-1, 1), labels.reshape(-1, 1), ids.reshape(-1, 1)))
    return new_pts

def read_lidar_state(lidar_state):
    gps_pos = lidar_state[:, 5:8]
    navi_pos = lidar_state[:, 8:11]
    # only use state 4 (RTK high precision)
    gps_state = lidar_state[:, 11]
    # only use state 3 (navi high precision)
    position_state = lidar_state[:, 12]

    return gps_pos, navi_pos, gps_state, position_state

def point_in_bbox(pts, bbox, eps=1e-2):
    pts_xyz = pts[:, :3].copy()

    rot = bbox[-1]

    rot_m = np.array([
        np.cos(rot), -np.sin(rot),
        np.sin(rot), np.cos(rot)
    ]).reshape(2, 2)

    local_xyz = pts_xyz - bbox[np.newaxis, :3]
    local_xyz[:, :2] = np.dot(local_xyz[:, :2], rot_m.T)

    w, l, h = bbox[3:6] + eps

    in_flag = (local_xyz[:, 0] > -w / 2.) & (local_xyz[:, 0] < w / 2.) & \
            (local_xyz[:, 1] > -l / 2.) & (local_xyz[:, 1] < l / 2.) & \
            (local_xyz[:, 2] > -h / 2.) & (local_xyz[:, 2] < h / 2.)
    return in_flag

def labeling_point_in_bbox(pts, gt_bboxes, gt_labels):
    pts_semantic_mask = np.zeros(pts.shape[0]).astype(np.int32)
    pts_instance_mask = -np.ones(pts.shape[0]).astype(np.int32)

    num_bbox = gt_bboxes.shape[0]
    for i in range(num_bbox):
        bbox_label = gt_labels[i]
        semantic_label = POINTS_CLASS_NAME.index(LABEL_NAME_MAPPING[BBOX_CLASS_NAME[bbox_label]])

        in_flag = point_in_bbox(pts, gt_bboxes[i])
        pts_semantic_mask[in_flag] = semantic_label
        pts_instance_mask[in_flag] = i
    return pts_semantic_mask, pts_instance_mask

def find_wrong_label_from_dl(pts, labels, bboxes, conf_thres, idx):
    is_need_label = False
    pts_num = len(pts)
    pts_semantic_mask = np.zeros(pts_num).astype(np.int32)
    pts_instance_mask = -np.ones(pts_num).astype(np.int32)
    obj_dict = {}
    obj_pt_list = []
    num_bbox = bboxes.shape[0]
    for i in range(num_bbox):
        if bboxes[i][-1] < conf_thres:
            continue
        bbox_label = int(bboxes[i][0])
        semantic_label = POINTS_CLASS_NAME.index(LABEL_NAME_MAPPING[BBOX_CLASS_NAME[bbox_label]])

        in_flag = point_in_bbox(pts, bboxes[i][1:8])
        obj_pt_list.append(sum(in_flag))
        pts_semantic_mask[in_flag] = semantic_label
        pts_instance_mask[in_flag] = i
    diff_num = 0
    for i in range(pts_num):
        bbox_label = pts_semantic_mask[i]
        if bbox_label == 0:
            continue
        seg_label = labels[i]

        if seg_label != bbox_label:
            if pts_instance_mask[i] not in obj_dict:
                obj_dict[pts_instance_mask[i]] = 1
            else:
                obj_dict[pts_instance_mask[i]] += 1
            diff_num += 1

    for item in obj_dict:
        diff_ratio = obj_dict[item] / obj_pt_list[item]
        if diff_ratio > 0.8:
            is_need_label = True
            break
        print(item, ':', diff_ratio)

    return is_need_label

def find_wrong_label_from_kdtree(pts, idx):
    is_need_label = False
    kt = spt.KDTree(data=pts[:, :3], leafsize=10)
    for i in range(len(pts)):
        if POINTS_CLASS_NAME[int(pts[i][3])] not in POINTS_BACKGROUND_NAME:
            continue
        distance_list, idx_list = kt.query(pts[i][:3], 10)
        if sum(distance_list) == 0:
            continue
        
        a = 1
    return is_need_label


def main():
    input_bin_folder = os.listdir(INPUT_BIN_FOLDER)
    input_label_folder = os.listdir(INPUT_LABEL_FOLDER)
    input_bbox_folder = os.listdir(INPUT_BBOX_FOLDER)
    # assert(len(input_bin_folder) == len(input_label_folder))
    bin_files = np.sort(input_bin_folder)
    label_files = np.sort(input_label_folder)
    bbox_files = np.sort(input_bbox_folder)
    lidar_state = np.loadtxt(INPUT_LIDAR_STATE, dtype=np.str)

    # assert(len(lidar_state) == len(input_label_folder))

    gps_pos, navi_pos, gps_state, position_state = read_lidar_state(lidar_state)
    start_veh_pose = [float(navi_pos[0][0]), float(navi_pos[0][1]), float(navi_pos[0][2])]
    frame_quary = []
    need_label_idx = []
    for i in tqdm(range(len(bin_files))):
        if position_state[i] == '3':
            veh_pose = [float(navi_pos[i][0]), float(navi_pos[i][1]), float(navi_pos[i][2])]
        else:
            continue

        bin_file_path = os.path.join(INPUT_BIN_FOLDER, bin_files[i])
        label_file_path = os.path.join(INPUT_LABEL_FOLDER, label_files[i])
        bbox_file_path = os.path.join(INPUT_BBOX_FOLDER, bbox_files[i])
        pts = np.fromfile(bin_file_path, dtype=np.float32)
        pts = pts.reshape(-1,8)[:, :4]
        labels = np.fromfile(label_file_path, dtype=np.int32)
        bboxes = np.loadtxt(bbox_file_path, dtype=np.float32)
        bboxes = bboxes[:, [0, 1, 2, 3, 5, 6, 4, 7, 9]]

        is_need_label = find_wrong_label_from_dl(pts, labels, bboxes, 0.8, str(i).zfill(6))
        if is_need_label:
            need_label_idx.append(i)

        new_pts = cal_relative_pose(pts, labels, veh_pose, start_veh_pose, i)
        frame_quary.append(new_pts.tolist())
        if len(frame_quary) > USE_HISTORY_FRAME:
            frame_quary = frame_quary[1:]
            all_pts  = []
            for j in range(USE_HISTORY_FRAME):
                all_pts += frame_quary[j]
        else:
            continue

        is_need_label = find_wrong_label_from_kdtree(np.array(all_pts), str(i).zfill(6))
        # if len(all_pts) == 0:
        #     all_pts = new_pts
        # else:
        #     all_pts = np.vstack((all_pts, new_pts))

        # save_all_pts = np.array(all_pts, dtype=np.float32)
        # save_name = os.path.join('/data/code/seg_data_tools/test/', str(i).zfill(6) + '.bin')
        # save_all_pts.tofile(save_name)
        a = 1

if __name__ == '__main__':
    main()