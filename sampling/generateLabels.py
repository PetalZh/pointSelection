import torch
from torch.utils.data import Dataset
import json
from models.pointnet import PointNetSeg
from kitti_utils import kitti_util
from kitti_utils.kitti_util import Calibration
import numpy as np
import os

def genLabel():
    label_dict = {}
    count = 7000
    for i in range(7000, 7481): #7481
        formatted_number = f"{i:06}"
        path = '/data/kitti_detection_3d/training/velodyne_reduced/{}.bin'.format(formatted_number)
        label_path = '/data/kitti_detection_3d/training/label_2/{}.txt'.format(formatted_number) #label_2, velodyne

        points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1,4])

        labels = getLabels(points, label_path)
        label_dict[formatted_number] = labels

        count += 1
        if count % 200 == 0:
            print(str(count))

        if count % 1000 == 0:
            write_to_file(label_dict, count)
            label_dict.clear()

    write_to_file(label_dict, count)

def write_to_file(label_dict, part):
    output_path = 'labels{}.json'.format(part)
    if not os.path.exists(output_path):
            f = open(output_path, "x")
    json.dump(label_dict, open(output_path, 'w'))

def getLabels(points, label_path):
    calib_path = label_path.replace('label_2', 'calib')
    labels = kitti_util.read_label(label_path)
    calib = Calibration(calib_path)
    output_labels = []
    
    for point in points:
        check = False
        for item in labels:
            if item.type == 'DontCare':
                continue
            _, box3d_pts_3d = kitti_util.compute_box_3d(item, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)

            # print(box3d_pts_3d_velo)
            min_values = np.min(box3d_pts_3d_velo, axis=0)
            max_values = np.max(box3d_pts_3d_velo, axis=0)

            min_x = min_values[0]
            max_x = max_values[0]

            min_y = min_values[1]
            max_y = max_values[1]

            min_z = min_values[2]
            max_z = max_values[2]

            if point[0] >= min_x and point[0] <= max_x and point[1] >= min_y and point[1] <= max_y and point[2] >= min_z and point[2] <= max_z:
                check = True
                break
        if check:
            output_labels.append(1)
        else:
            output_labels.append(0)

    return output_labels

genLabel()