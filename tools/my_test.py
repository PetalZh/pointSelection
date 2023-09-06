from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.special import softmax
import torch
import time
from numpy import genfromtxt
import open3d as o3d
from mmdet3d.datasets.transforms.kitti_utils import kitti_util
from mmdet3d.datasets.transforms.kitti_utils.kitti_util import Calibration

max_distance = 0
def sim(x, y):
    # print('shape of x: ', x)
    # print('shape of y: ', y)
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    # print('shape of x: ', x)
    # print('shape of y: ', y)
    # x.unsqueeze(0)
    eu_dist = euclidean_distances(x,y)
    cosine_sim = cosine_similarity(x, y)
    # print(eu_dist)
    # print('max_distance', max_distance)

    n_cosine_sim = 0.5 * (cosine_sim + 1)

    return 0.5 * eu_dist + 0.5 * cosine_sim

def distance_metric(X):
    # print("shape of X: ", X.shape)
    global max_distance
    max_distance = np.max(euclidean_distances(X[:, :4]))
    return pairwise_distances(X, metric=sim)

def test_torch_cat():
    points1 = torch.tensor(np.ones((10, 4)))
    points2 = torch.tensor(np.ones((10, 498)) * 2)

    print(points1)
    print(points2)

    result = torch.cat((points1, points2), 1)

    print(result)

def visualization():
    # path = 'pc.csv'
    # points = genfromtxt(path, delimiter=',')
    # print(points)

    points = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    # colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

    
# # visualization()

# points = np.random.rand(1000, 20)

# # hier_cluster = AgglomerativeClustering(n_clusters=8, metric=distance_metric, linkage='average')
# # hier_cluster.fit(points)

# time_start = time.time()
# # precompute_distance = distance_metric(points)
# # print('precompute distance done')
# time_stop1 = time.time()
# hier_cluster = AgglomerativeClustering(n_clusters=8, linkage='average')
# hier_cluster.fit(points)
# # hier_cluster = AgglomerativeClustering(n_clusters=8, metric=distance_metric, linkage='average')
# # hier_cluster.fit(precompute_distance)
# time_stop2 = time.time()

# time_precompute = (time_stop1 - time_start)/60
# time_all = (time_stop2 - time_start)/60
# print('time precompute: ', str(time_precompute))
# print('time total: ', str(time_all))

# # print(hier_cluster.labels_)
# # print(hier_cluster.children_)


# # score = [1, 2, 3, 3, 4]
# # print(softmax(score))
# # print(score.sum())
# # print(softmax(score).sum())

# # test_torch_cat()

from waymo_open_dataset.protos.metrics_pb2 import Objects
from waymo_open_dataset import protos

def checkBin():
    path_gt = '/data/waymo/waymo_format/gt.bin'
    path_result = '/home/xiaoyu/experiments/mmdetection3d/tools/waymo_results.bin'

    # pc = np.fromfile(path_result, dtype=np.float32, count=-1)
    
    # get timestamp list in result.bin
    with open(path_result, 'rb') as f:
        # print(f.read())
        objs = Objects()
        objs.ParseFromString(f.read())
    
    frame_set = set()
    for obj in objs.objects:
        # print(obj.frame_timestamp_micros)
        frame_set.add(obj.frame_timestamp_micros)

    filtered_gt_obj = getFilteredGT(frame_set)

    output_path = '/home/xiaoyu/experiments/mmdetection3d/tools/gt_subset.bin'
    with open(output_path,'wb') as f:
        f.write(filtered_gt_obj.SerializeToString())
    


def getFilteredGT(frame_set):
    output_obj = Objects()
    path_gt = '/data/waymo/waymo_format/gt.bin'

    with open(path_gt, 'rb') as f:
        # print(f.read())
        objs = Objects()
        objs.ParseFromString(f.read())
    
    for obj in objs.objects:
        # print(obj.frame_timestamp_micros)
        if obj.frame_timestamp_micros in frame_set:
            output_obj.objects.append(obj)
    
    print('length of before {}, length of after {}'.format(len(objs.objects), len(output_obj.objects)))

    return output_obj
    

# checkBin()

from nuscenes.nuscenes import NuScenes

def nuscenes():
    # version='v1.0-trainval', dataroot='/data/nuscenes/full', verbose=True
    nusc = NuScenes(version='v1.0-mini', dataroot='/data/nuscenes/mini', verbose=True)
    # nusc.list_scenes()

    cate_dict = {} #{cate:[num_of_points, num_of_item]}

    for scene in nusc.scene:
        sample_token1 = scene['first_sample_token']
        sample_token2 = scene['last_sample_token']

        # token_list = []
        # token_list.append(sample_token1)
        # token_list.append(sample_token2)

        for sample_token in [sample_token1, sample_token2]:
            sample = nusc.get('sample', sample_token)
            anna_tokens = sample['anns']
            for anna_token in anna_tokens:
                anna_metadata =  nusc.get('sample_annotation', anna_token)
                cate_name = anna_metadata['category_name']
                num = anna_metadata['num_lidar_pts']

                if cate_name in cate_dict.keys():
                    cate_dict[cate_name][0] += num
                    cate_dict[cate_name][1] += 1
                else:
                    cate_dict[cate_name] = [num, 1]
    print(cate_dict)

    total_obj_points = 0
    for key in cate_dict.keys():
        total_obj_points += cate_dict[key][0]

    print('total_obj_points: ', total_obj_points)

# nuscenes()