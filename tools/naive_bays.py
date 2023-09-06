from mmdet3d.datasets.transforms.kitti_utils import kitti_util
from mmdet3d.datasets.transforms.kitti_utils.kitti_util import Calibration
from numpy import genfromtxt
import numpy as np
import os
import json

def getPoints(): 
    sta_dict = {}
    counter = 0
    for i in range(1, 7481): #7481
        formatted_number = f"{i:06}"
        path = '/data/kitti_detection_3d/training/velodyne_reduced/{}.bin'.format(formatted_number)
        label_path = '/data/kitti_detection_3d/training/label_2/{}.txt'.format(formatted_number) #label_2, velodyne

        points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1,4])
        # print(points.shape)
        x_range = getRangeList(points, 0, 80)
        y_range = getRangeList(points, 0, 60)

        getStatistics(x_range, y_range, label_path, sta_dict)

        counter += 1
        if counter % 100 == 0:
            print(str(counter))

        if counter % 1000 == 0:
            write_to_file(sta_dict, counter)
            sta_dict.clear()

    # print(sta_dict)
    write_to_file(sta_dict, counter)


def write_to_file(sta_dict, part):
    output_path = '/home/xiaoyu/experiments/mmdetection3d/tools/output/bays_{}.json'.format(part)
    if not os.path.exists(output_path):
            f = open(output_path, "x")
    json.dump(sta_dict, open(output_path, 'w'))

def getRangeList(points, dim, bin_num):

    count, bins_edge = np.histogram(points[:, dim], bins = bin_num)

    pdfs = count / sum(count)

    output = [] #[[range_start, range_end], pdf]
    for (i, pdf) in enumerate(pdfs):
        range = [bins_edge[i], bins_edge[i+1]]
        record = [range, pdf]
        output.append(record)
    return output

def getStatistics(x_range, y_range, label_path, sta_dict):
    # sta_dict: {ij:{obj_x:[], obj_y: []}}
    # x_item: [[x_start, x_end], x_pdf]
    for (i, x_item) in enumerate(x_range):
        for (j, y_item) in enumerate(y_range):
            cube_key = '{}_{}'.format(i, j)
            if checkItemExistance(x_item[0], y_item[0], label_path):
                addPDFtoDict(cube_key, sta_dict, x_item[1], y_item[1], True)
            else:
                # pass
                addPDFtoDict(cube_key, sta_dict, x_item[1], y_item[1], False)


def addPDFtoDict(cube_key, sta_dict, pdf_x, pdf_y, isObj):
    if cube_key not in sta_dict.keys():
        if isObj:
            sta_dict[cube_key]= {'obj_x':[pdf_x], 'obj_y':[pdf_y], 'env_x':[], 'env_y':[]}
        else:
            sta_dict[cube_key]= {'obj_x':[], 'obj_y':[], 'env_x':[pdf_x], 'env_y':[pdf_y]}
    else:
        if isObj:
            sta_dict[cube_key]['obj_x'].append(pdf_x)
            sta_dict[cube_key]['obj_y'].append(pdf_y)
        else:
            sta_dict[cube_key]['env_x'].append(pdf_x)
            sta_dict[cube_key]['env_y'].append(pdf_y)


def checkItemExistance(x_range, y_range, label_path):
    calib_path = label_path.replace('label_2', 'calib')
    # print(label_path)
    # print(calib_path)
    labels = kitti_util.read_label(label_path)
    calib = Calibration(calib_path)
    
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

        # min_z = min_values[2]
        # max_z = max_values[2]

        if x_range[0] > max_x or x_range[1] < min_x:
            # check = False
            continue
        elif y_range[0] > max_y or y_range[1] < min_y:
            # check = False
            continue
        else:
            check = True
            break
    return check

def getPC_and_label():
    random_integers = np.random.randint(1, 7481, size=100)

    for idx, number in enumerate(random_integers): #7481
        formatted_number = f"{number:06}"
        path = '/data/kitti_detection_3d/training/velodyne_reduced/{}.bin'.format(formatted_number)
        label_path = '/data/kitti_detection_3d/training/label_2/{}.txt'.format(formatted_number) #label_2, velodyne

        points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1,4])
        labels = getLabels(label_path)

        np.save('aaa/pc/pc_{}'.format(idx), points)
        np.save('aaa/label/label_{}'.format(idx), labels)





def getLabels(label_path):
    calib_path = label_path.replace('label_2', 'calib')
    # print(label_path)
    # print(calib_path)
    labels = kitti_util.read_label(label_path)
    calib = Calibration(calib_path)
    
    check = False
    label_list = []
    for item in labels:
        if item.type == 'DontCare':
            continue
        _, box3d_pts_3d = kitti_util.compute_box_3d(item, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)

        label_list.append(box3d_pts_3d_velo.tolist())
    
    return np.array(label_list)



# getPoints()
getPC_and_label()