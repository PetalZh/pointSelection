from mmdet3d.structures.points import BasePoints, get_points_type, LiDARPoints
from .my_sample_method import MySample
from .my_sample_baselines import SampleBaselines
from .kitti_utils import kitti_util
from .kitti_utils.kitti_util import Calibration

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils import constants
import os
import json
import time
import torch
import numpy as np

def test():
    print('testestestestest')

def doSample(points, sample_method, sample_rate, pts_filename, oc_node_capacity=200, octree_height=10):
    # my sample method
        constants.counter += 1
        # np.savetxt('pc_sample_before_0.15{}.csv'.format(constants.counter), points, delimiter=",")

        # print('sample method: {}, sample_rate{}, oc_capacity: {}'.format(self.sample_method, self.sample_rate, self.oc_node_capacity))
        # print('before sample: ', points.shape)

        # instance_points_before = self.getInstancePoints(self.pts_filename, points)
        # print('before {}'.format(len(points)))

        # print('method: {}'.format(sample_method))
        # print('before {}'.format(points.shape[0]))
        
        if sample_method == 'feat_octree':
            # print(constants.feature_model)
            if constants.feature_model == 'pointnet':
                # print(constants.feature_model)
                my_sample = MySample(points, sample_rate, oc_node_capacity, 'pointnet')
                featured_points = my_sample.getFeature(points)
                # print('feature point shape: ', featured_points.shape)
            elif constants.feature_model == 'pointnet2':
                
                my_sample = MySample(points, sample_rate, oc_node_capacity, 'pointnet2')
                time_start = time.time()
                featured_points = my_sample.getFeature_Pointnet2(points)
                time_end = time.time()
                print('time: {}s'.format(time_end - time_start))
                # print('feature point shape: ', featured_points.shape)
            elif constants.feature_model == 'spvnas':
                my_sample = MySample(points, sample_rate, oc_node_capacity, 'spvnas')
                
                featured_points = my_sample.getFeature_spvnas(points)
                
            elif constants.feature_model == 'spvcnn':
                my_sample = MySample(points, sample_rate, oc_node_capacity, 'spvcnn')
                featured_points = my_sample.getFeature_spvcnn(points)
    
            my_sample.feat_octree(featured_points)
 
            # self.getSimilarities(my_sample.leaves)
            # np.savez('kmeans_spvnas_cluster_{}.npz'.format(constants.counter), *(my_sample.leaves))

            # print('total points number: ', featured_points.shape[0])
            # self.getStatistics(my_sample.leaves)
            # print('before sample', len(points))
             # np.savetxt('pc_before_0.15{}.csv'.format(constants.counter), points, delimiter=",")np.array(my_sample.leaves)
            
            
            points = my_sample.getPoints3(points.shape[0])

            # points = my_sample.getPoints2(points.shape[0])

            # points = my_sample.getSampledPoints(points.shape[0])
            # print('after sample', len(points))

        if sample_method == 'octree':
            baselines = SampleBaselines(points, sample_rate)
            # print('total points number: ', points.shape[0])
            points = baselines.octree(points, octree_height)
            # self.getStatistics(baselines.oc_node_list)
        if sample_method == 'random':
            constants.total_points += len(points)
            # print(len(points))
            baselines = SampleBaselines(points, sample_rate)
            points = baselines.randomSample(points)
        if sample_method == 'fps':
            # print('before: {}'.format(points.shape[0]))
            baselines = SampleBaselines(points, sample_rate)
            points = baselines.fpsSample(points)
            # print('after: {}'.format(points.shape[0]))
        if sample_method == 'grid_fps':
            baselines = SampleBaselines(points, sample_rate)
            points = baselines.gridSample(points)
        if sample_method == 'statistic':
            time_start = time.time()

            my_sample = MySample(points, sample_rate, oc_node_capacity, 'spvnas')
            points = my_sample.statistic_sample(points)
            constants.total_points += len(points)
            time_end = time.time()
        if sample_method == 'statistic_nb':
            # print('before {}'.format(points.shape[0]))
            # np.savetxt('pc_before_{}.csv'.format(constants.counter), points, delimiter=",")
            if constants.counter == 1:
                constants.empty_keys = getEmptyGridKeys()
                constants.nb_dict = getNBDict()

            my_sample = MySample(points, sample_rate, oc_node_capacity, 'spvnas')  

            # print('before {}'.format(len(points)))
            points = my_sample.statistic_nb(points, pts_filename)
            # print('after {}'.format(len(points)))
            # np.savetxt('pc_after_{}.csv'.format(constants.counter), points, delimiter=",")
            # print('after {}'.format(points.shape[0]))
        # print('after {}'.format(points.shape[0]))
            # print('time: {}s'.format(time_end - time_start))

        # print('after {}'.format(len(points)))
            # original = points
            # points = self.getInstanceOnlyPointSet(self.pts_filename, points)
            # if len(points) < 100:
            #     points = original
            # time_end = time.time()
            # print('time: {}s'.format(time_end - time_start))
            # print('after sample: {} points'.format(points.shape[0]))


        # np.savetxt('pc_after_0.15{}.csv'.format(constants.counter), points, delimiter=",")
        # instance_points_after = self.getInstancePoints(self.pts_filename, points)
        # self.getInstanceStatistic(instance_points_before, instance_points_after)
        
        # print(constants.instance_point_dict['Car'])

        # print('before {}, after {}'.format(instance_points_before, instance_points_after))
        return points

def getEmptyGridKeys():
    path = '/home/xiaoyu/experiments/mmdetection3d/tools/nb_models/intersection.json'
    if os.path.exists(path):
        empty_keys = list(json.load(open(path)))
    return empty_keys
def getNBDict():
    #pc_preprocess
    #pc_preprocess_complement
    path = '/home/xiaoyu/experiments/mmdetection3d/tools/nb_models/pc_preprocess_categorical.json'
    if os.path.exists(path):
        nb_dict = dict(json.load(open(path)))
    return nb_dict

def getInstanceStatistic(self, instance_points_before, instance_points_after):
    for key in instance_points_before.keys():
        count_before = instance_points_before[key]
        count_after = instance_points_after[key]

        if key not in constants.instance_point_dict.keys():
            constants.instance_point_dict[key] = [0, 0]
            
        constants.instance_point_dict[key][0] += count_before
        constants.instance_point_dict[key][1] += count_after
    path = '/home/xiaoyu/experiments/mmdetection3d/tools/instance_sta.txt'
    path_count = '/home/xiaoyu/experiments/mmdetection3d/tools/instance_count.txt'
    
    # 3768
    if constants.counter == 3768:
        print('test write to file')
        if not os.path.exists(path):
            f = open(path, "x")
        json.dump(constants.instance_point_dict, open(path, 'w'))

        if not os.path.exists(path_count):
            f = open(path_count, "x")
        json.dump(constants.instance_count_dict, open(path_count, 'w'))

        for key in constants.instance_point_dict.keys():
            print('lower bound of the sampling:', int(constants.instance_point_dict[key][0]/3786))

        for key in constants.instance_count_dict.keys():
            x = constants.instance_xyz_dict[key][0]/constants.instance_count_dict[key]
            y = constants.instance_xyz_dict[key][1]/constants.instance_count_dict[key]
            z = constants.instance_xyz_dict[key][2]/constants.instance_count_dict[key]
            
            print('obj: {}, x: {}, y: {}, z: {}'.format(key, x, y, z))
        print('Total points: {}'.format(constants.total_points))

def getInstanceOnlyPointSet(self, pts_path, points):
        output = {}
        label_path = pts_path.replace('velodyne_reduced', 'label_2')
        label_path = label_path.replace('bin', 'txt')

        calib_path = label_path.replace('label_2', 'calib')
        # print(label_path)
        # print(calib_path)
        labels = kitti_util.read_label(label_path)
        calib = Calibration(calib_path)
        
        output = np.empty((0, 4), dtype=np.float32)
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

            
            for point in points:
                if point[0] >= min_x and point[0] <= max_x and point[1] >= min_y and point[1] <= max_y and point[2] >= min_z and point[2] <= max_z:
                    output = np.vstack((output, point))

        return output

def getInstancePoints(self, pts_path, points):
    output = {}
    label_path = pts_path.replace('velodyne_reduced', 'label_2')
    label_path = label_path.replace('bin', 'txt')

    calib_path = label_path.replace('label_2', 'calib')
    # print(label_path)
    # print(calib_path)
    labels = kitti_util.read_label(label_path)
    calib = Calibration(calib_path)
    
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

        if item.type in constants.instance_count_dict.keys():
            constants.instance_count_dict[item.type] += 1
        else:
            constants.instance_count_dict[item.type] = 1


        count = 0
        obj = np.empty((0, 4), dtype=np.float32)
        for point in points:
            if point[0] >= min_x and point[0] <= max_x and point[1] >= min_y and point[1] <= max_y and point[2] >= min_z and point[2] <= max_z:
                count += 1
                obj = np.vstack((obj, point))

        self.getInstanceUniqueXYZ(obj, item.type)

        if item.type in output.keys():
            output[item.type] += count
        else:
            output[item.type] = count
        
    # print(output)

    return output

def getInstanceUniqueXYZ(self, points, type):
    unique_x = len(np.unique(points[:, 0]))
    unique_y = len(np.unique(points[:, 1]))
    unique_z = len(np.unique(points[:, 2]))

    dict = constants.instance_xyz_dict
    if type in dict.keys():
        dict[type][0] += unique_x
        dict[type][1] += unique_y
        dict[type][2] += unique_z
    else:
        dict[type] = [unique_x, unique_y, unique_z]