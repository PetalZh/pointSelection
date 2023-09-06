import torch
import numpy as np
from .models.pointnet import PointNetSeg
from .models.pointnet2 import PointNet2SemSeg
from .models.classifier import Classifier
from .models.spvnas.feature import spvnarsFeature, spvcnnFeature
import torch.multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, BisectingKMeans
from torch_cluster import fps

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections, RandomBinaryProjectionTree
import itertools
import os
import json
import joblib


from scipy.special import softmax
import pdb
import math
# import hdbscan
from decimal import Decimal
from utils import constants
# from torch_cluster import fps
import statistics
import time
from scipy.signal import argrelextrema


class MySample():
    def __init__(self, pcd, sample_rate, oc_node_capacity, feature_model):
        self.pcd = pcd
        self.sample_rate = sample_rate
        self.oc_node_capacity = oc_node_capacity
        self.leaves = []
        self.feature_model = feature_model
        self.sim_thresh = 1

    def getFeature_spvnas(self, points):
        points_feature = spvnarsFeature(points)
        # print('torch tensor device: ', points_feature.get_device())
        points = torch.from_numpy(points).cuda()
        # print(points[[1, 10, 100, 500, 1000, 5000, 9000], :8])

        output = torch.cat((points, points_feature), 1)

        return output.detach().cpu().numpy()

    def getFeature_spvcnn(self, points):
        points_feature = spvcnnFeature(points)
        # print('torch tensor device: ', points_feature.get_device())
        points = torch.from_numpy(points).cuda()
        # print(points[[1, 10, 100, 500, 1000, 5000, 9000], :8])

        output = torch.cat((points, points_feature), 1)

        return output.detach().cpu().numpy()


    def getFeature_Pointnet2(self, points):
        checkpoints_path =  '/home/xiaoyu/experiments/mmdetection3d/checkpoints/pointnet/pointnet2-all.pth'
        model = PointNet2SemSeg(19, feature_dims = 1)

        model = torch.nn.DataParallel(model)
        model.cuda()
        model.load_state_dict(torch.load(checkpoints_path))

        points = torch.from_numpy(points[:, :4]).float().transpose(1, 0).cuda()
        points_inputs = points.clone()
        points_inputs = points_inputs.unsqueeze(0) # shape (1, 4, N)
        with torch.no_grad():
            model.eval()
            pred, intermediate_outputs = model(points_inputs)
            points_feature = intermediate_outputs[0].squeeze().transpose(1, 0).cuda() #shape (N, 128)
            # max pool to (N, 16)
            points_feature = points_feature.reshape((points_feature.shape[0], 16, 8)) 
            points_feature, _ = torch.max(points_feature, dim=2)

            # print(points_feature[[1, 10, 100, 500, 1000, 5000, 9000], :8])
            
            trans_points = points.transpose(1, 0)
            output = torch.cat((trans_points, points_feature), 1) #shape: (N, 4 + 16)
        return output.detach().cpu().numpy() #points.detach().cpu().numpy()

    def getFeature(self, points):
        checkpoints_path =  '/home/xiaoyu/experiments/mmdetection3d/checkpoints/pointnet/pointnet-all.pth'
        model = PointNetSeg(19, input_dims = 4, feature_transform=True)

        model = torch.nn.DataParallel(model)
        model.cuda()
        model.load_state_dict(torch.load(checkpoints_path))

        points = torch.from_numpy(points[:, :4]).float().transpose(1, 0).cuda() #shape (4, N)
        points_inputs = points.clone()
        points_inputs = points_inputs.unsqueeze(0) # shape (1, 4, N)
        with torch.no_grad():
            model.eval()
            pred, intermediate_outputs = model(points_inputs)
            # print("testestestest", intermediate_outputs[0].shape)
            points_feature = intermediate_outputs[0].squeeze().transpose(1, 0).cuda() #shape (N, 128)
            
            # max pool (N, 16) 
            points_feature = points_feature.reshape((points_feature.shape[0], 16, 8)) 
            points_feature, _ = torch.max(points_feature, dim=2)

            # print('shape of reshape: ', points_feature.shape) 
            # print(points_feature)

            trans_points = points.transpose(1, 0)

            output = torch.cat((trans_points, points_feature), 1) #shape: (N, 4 + 16)
        return output.detach().cpu().numpy() #points.detach().cpu().numpy()

    #sim_thresh=0.995
    def feat_octree(self, points, sim_thresh=0.95, method='hierarchical'):
        # print('sim thresh:  ', str(sim_thresh))
        self.sim_thresh = sim_thresh
        if points.shape[0] > 1000:
            clusters = self.getOctreeSplitFor4(points)

            for cluster in clusters:
                self.feat_octree(cluster)
        else:
            if points.shape[0] <= self.oc_node_capacity: #self.oc_node_capacity
                if points.shape[0] != 0:
                    self.leaves.append(points)
                return
            else:
                similarity_matrix = cosine_similarity(points[:, 5:]) # euclidean_distances(points[:, 4:])
                overall_similarity = similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)].mean()
                # print(overall_similarity)
                # if overall_similarity < 1:
                #     print("overall_similarity: ", overall_similarity)
                # print("overall_similarity: ", overall_similarity)
                
                if overall_similarity > sim_thresh:
                    self.leaves.append(points)
                    return
            if method == 'kmeans':
                cluster = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(points)
                cluster_label = cluster.labels_
            elif method == 'hierarchical':
                cluster = AgglomerativeClustering(n_clusters=8, linkage='complete').fit(points)
                cluster_label = cluster.labels_
                # precompute_distance = distance_metric(points)
                # hier_cluster = AgglomerativeClustering(n_clusters=8, metric='precomputed', linkage='complete')
                # hier_cluster.fit(precompute_distance)#[:, :4]
            elif method == 'classifier':
                cluster_label = self.getClassifyResults(points)
   
            cluster_dict = self.getClusterDict(cluster_label, points)
            for key, value in cluster_dict.items():
                cluster = np.vstack(value)
                self.feat_octree(cluster)

    def getClusterDict(self, labels, points):
        cluster_dict={}

        for index, item in enumerate(labels):  #dbscan.labels_, 
            if item in cluster_dict.keys():
                cluster_dict[item].append(points[index, :])
            else:
                c_list = []
                c_list.append(points[index, :])
                cluster_dict[item] = c_list
        return cluster_dict

    def getClassifyResults(self, points):
        checkpoints_path = '/home/xiaoyu/experiments/mmdetection3d/checkpoints/classifier/model_epoch_40.pth'
        
        points = torch.from_numpy(points).cuda()

        model = Classifier(8)
        model = torch.nn.DataParallel(model)
        model.cuda()
        model.load_state_dict(torch.load(checkpoints_path))

        points = points.unsqueeze(0)
        # print('input point shape: ', points.shape)

        with torch.no_grad():
            model.eval()
            output = model(points)

            output = output[:, :, 3].cpu().numpy().tolist()
            print(output)

        return output[0]


    def getSoftmax(self):
        #np.empty((0, 4), dtype=np.float32)
        scores = []
        for cluster in self.leaves:
            # calculate volume/number of points and append to scores
            score = float(Decimal(self.getVolume(cluster)/cluster.shape[0]))
            # score = self.getVolume(cluster)/cluster.shape[0]
            scores.append(score)

            # print(scores)
        
        # softmax socre
        softmax_scores = softmax(np.array(scores))
        # print(softmax_scores)
        return softmax_scores

    def getPoints3(self, total_points):
        output = []
        sta_size = []
        sta_sim = []
        for item in self.leaves:
            sta_size.append(len(item))
            if len(item) == 1:
                overall_similarity = 0
            else:
                similarity_matrix = cosine_similarity(item[:, 5:]) 
                overall_similarity = similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)].mean()
            
            sta_sim.append(overall_similarity)

        mean = statistics.mean(sta_size)
        std = statistics.stdev(sta_size)
        normal_dist = torch.distributions.Normal(mean, std)

        for item in self.leaves:
            prob = normal_dist.log_prob(torch.tensor(len(item))).exp()
            
            reward = prob * 100 - 0.5
            rate = self.sample_rate + reward * (1 - self.sample_rate)
            rate = rate if rate > 0 else self.sample_rate * 0.1

            # if constants.counter == 7:
            # print('mean {}, std {}, number of points: {}, prob: {}'.format(mean, std, len(item), prob.item()))
            #     print('number of points: {}, rate: {}'.format(len(item), rate))
            
            # similarity_matrix = cosine_similarity(item[:, 5:]) # euclidean_distances(points[:, 4:])
            # overall_similarity = similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)].mean()

            # if overall_similarity < 0.8:
            #     rate = rate * 0.5

            # if overall_similarity > 0.95:
            #     rate = rate * 0.5

            sample_length = int(np.ceil(rate * item.shape[0]))
            choice = np.random.choice(item.shape[0], sample_length, replace=False)
            output.append(item[choice][:, :4])


        for i, item in enumerate(output):
            overall_similarity = sta_sim[i]
            mean = np.mean(sta_sim)
            std = np.std(sta_sim)

            if overall_similarity < mean - 2 * std or overall_similarity > mean + 2 * std:
                rate = self.sample_rate
                sample_length = int(np.ceil(rate * item.shape[0]))
                choice = np.random.choice(item.shape[0], sample_length, replace=False)
                item[choice]

        
        output = np.vstack(output)
        
        sample_length = np.ceil(total_points * self.sample_rate)

        # print('sample length: {}, expected length: {}'.format(output.shape, sample_length))

        choice = np.random.choice(len(output), int(sample_length), replace=True)

        return output

    def getPoints2(self, total_points):
        output = []
        # print('before sample: ', total_points)
        for item in self.leaves:
            if len(item) >= 400:
                sample_rate = self.sample_rate * 0.3
            if len(item) > 300 and len(item) < 400:
                sample_rate = self.sample_rate * 0.3
            if len(item) > 200 and len(item) < 300:
                sample_rate = self.sample_rate * 0.7
            if len(item) > 100 and len(item) < 200:
                sample_rate = self.sample_rate + 0.3
            if len(item) > 50 and len(item) < 100:
                sample_rate = self.sample_rate + 0.3
            else:
                sample_rate = self.sample_rate

            sample_length = int(np.ceil(sample_rate * item.shape[0]))
            choice = np.random.choice(item.shape[0], sample_length, replace=True)
            output.append(item[choice][:, :4])
        output = np.vstack(output)
        # print('after sample: ', output.shape)

        sample_length = np.ceil(total_points * self.sample_rate)
        choice = np.random.choice(len(output), int(sample_length), replace=True)
        
        # print('total: {}, selected: {}, expected: {}'.format(total_points, output.shape, sample_length))

        return output[choice]

    def getPoints(self, total_points):
        output = []
        while(len(self.leaves) != 0):
            softmax_scores = self.getSoftmax()
            sorted_indices = np.argsort(-softmax_scores)
            # print(sorted_indices)
            sorted_softmax_score = softmax_scores[sorted_indices]

            pop_list = []
            for i, score in enumerate(sorted_softmax_score):
                suggested_length = int(np.ceil(score * self.sample_rate * total_points))
                # print('total number of leaf', len(self.leaves))
                # print('leaf index', sorted_indices[i])
                if suggested_length > len(self.leaves[sorted_indices[i]])*0.5:
                    # sample_length = np.ceil(self.leaves[sorted_indices[i]].shape[0] * self.sample_rate)
                    # choice = np.random.choice(self.leaves[sorted_indices[i]].shape[0], int(sample_length), replace=False)
                    # output.append(self.leaves[sorted_indices[i]][:, :4])
                    leaf = self.leaves[sorted_indices[i]]
                    if(len(leaf) < 20):
                        output.append(leaf[:, :4])
                    else:
                        sample_length = np.ceil(leaf.shape[0] * (self.sample_rate + 0.3))
                        choice = np.random.choice(leaf.shape[0], int(sample_length), replace=False)
                        output.append(leaf[:, :4])#leaf[choice][:, :4]
                    pop_list.append(sorted_indices[i])

                    pop_list.sort(reverse=True)
                    for index in pop_list:
                        self.leaves.pop(index)
                    break
                else:
                    sample_length = np.ceil(self.leaves[sorted_indices[i]].shape[0] * self.sample_rate)
                    choice = np.random.choice(self.leaves[sorted_indices[i]].shape[0], int(sample_length), replace=False)
                    output.append(self.leaves[sorted_indices[i]][choice][:, :4])
                    pop_list.append(sorted_indices[i])

                    if len(pop_list) == len(self.leaves):
                        self.leaves = []
                        break
        output  = np.vstack(output)
        sample_length = np.ceil(total_points * self.sample_rate)
        choice = np.random.choice(len(output), int(sample_length), replace=False)

        output = output[choice]
        return output

    def getSampledPoints(self, total_points):
        softmax_scores = self.getSoftmax()
        output = []
        for index, item in enumerate(self.leaves):
            # get the number of points need to be sampled
            
            # sample_length_soft = int(np.ceil(softmax_scores[index] * self.sample_rate * total_points))
            
            # if item.shape[0] < sample_length_soft:
            #     print('score {}, rate {}, points {}'.format(softmax_scores[index], self.sample_rate, total_points))
            #     print('with {} points, total sample {} '.format(item.shape[0], sample_length_soft))


            # sample_length = int(np.ceil(softmax_scores[index] * self.sample_rate * total_points))
            sample_length = int(np.ceil(self.sample_rate * item.shape[0]))
            choice = np.random.choice(item.shape[0], sample_length, replace=False)
            # print("type of output: " , type(output))
            output.append(item[choice][:, :4])
        return np.vstack(output)

    def getVolume(self, points):
        max_index = np.argmax(points[:, :3], axis=0)
        min_index = np.argmin(points[:, :3], axis=0)

        xmax = points[max_index[0]][0]
        ymax = points[max_index[1]][1]
        zmax = points[max_index[2]][2]

        xmin = points[min_index[0]][0]
        ymin = points[min_index[1]][1]
        zmin = points[min_index[2]][2]

        volume = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)

        return volume

    def getOctreeSplitFor4(self, points):
        # print(points.shape)
        #tensor_pcd = torch.from_numpy(points[:, [0,1,2]])

        max_index = np.argmax(points[:, :3], axis=0)
        min_index = np.argmin(points[:, :3], axis=0)

        # Compute the size of each cluster
        xmax = points[max_index[0]][0]
        ymax = points[max_index[1]][1]

        xmin = points[min_index[0]][0]
        ymin = points[min_index[1]][1]

        xsize = (xmax - xmin) / 2
        ysize = (ymax - ymin) / 2

        # Assign each point to a cluster
        clusters = []
        for i in range(4):
            xmin_cluster = xmin + i % 2 * xsize
            ymin_cluster = ymin + (i // 2) % 2 * ysize

            xmax_cluster = xmin_cluster + xsize
            ymax_cluster = ymin_cluster + ysize

            mask = (points[:, 0] >= xmin_cluster) & (points[:, 0] < xmax_cluster) & (points[:, 1] >= ymin_cluster) & (points[:, 1] < ymax_cluster)
            clusters.append(points[mask])

        # Print the number of points in each cluster
        # for i, cluster in enumerate(clusters):
        #     print(f"Cluster {i+1}: {cluster.shape[0]} points")

        return clusters

    def getOctreeSplit(self, points):
        # print(points.shape)
        #tensor_pcd = torch.from_numpy(points[:, [0,1,2]])

        max_index = np.argmax(points[:, :3], axis=0)
        min_index = np.argmin(points[:, :3], axis=0)

        # Compute the size of each cluster
        xmax = points[max_index[0]][0]
        ymax = points[max_index[1]][1]
        zmax = points[max_index[2]][2]

        xmin = points[min_index[0]][0]
        ymin = points[min_index[1]][1]
        zmin = points[min_index[2]][2]

        xsize = (xmax - xmin) / 2
        ysize = (ymax - ymin) / 2
        zsize = (zmax - zmin) / 2

        # Assign each point to a cluster
        clusters = []
        for i in range(8):
            xmin_cluster = xmin + i % 2 * xsize
            ymin_cluster = ymin + (i // 2) % 2 * ysize
            zmin_cluster = zmin + i // (2 ** 2) * zsize
            xmax_cluster = xmin_cluster + xsize
            ymax_cluster = ymin_cluster + ysize
            zmax_cluster = zmin_cluster + zsize
            mask = (points[:, 0] >= xmin_cluster) & (points[:, 0] < xmax_cluster) & (points[:, 1] >= ymin_cluster) & (points[:, 1] < ymax_cluster) & (points[:, 2] >= zmin_cluster) & (points[:, 2] < zmax_cluster)
            clusters.append(points[mask])

        # Print the number of points in each cluster
        # for i, cluster in enumerate(clusters):
        #     print(f"Cluster {i+1}: {cluster.shape[0]} points")

        return clusters

    def getChoice(self, choice_indices, target_list):
        output = []
        for idx in choice_indices:
           output.append(target_list[idx])
        return output

    # input: clusters, output: sampled points
    def LSH(self, cluster, exp_p):
        query = np.mean(np.array(cluster), axis=0)
        num_projections = 10
        dim = len(cluster[0])
        rbpt = RandomBinaryProjections('rbpt', num_projections)
        engine = Engine(dim, lshashes=[rbpt])
        for i, vector in enumerate(cluster):
            engine.store_vector(vector, '%d' % i)
        nn_list = engine.neighbours(query)

        output = np.empty((0, 4), dtype=np.float32)

        nn_idx = [int(item[1]) for item in nn_list]
        # print(nn_idx)

        if exp_p > len(nn_list):
            exp_p = exp_p - len(nn_list)

            output = np.vstack((output, cluster[nn_idx]))

            cluster_idx = [num for num in range(0, len(cluster))]

            left_idx = [item for item in cluster_idx if item not in nn_idx]

            choice = np.random.choice(len(left_idx), exp_p, replace=False)
            left_idx = self.getChoice(choice, left_idx)
            output = np.vstack((output, cluster[left_idx]))
        else:
            choice = np.random.choice(len(nn_idx), exp_p, replace=False)
            nn_idx = self.getChoice(choice, nn_idx)
            output =  np.vstack((output, cluster[nn_idx]))
        return output        
    
    def FPS(self, pcd, exp_p):
        tensor_pcd = torch.from_numpy(pcd[:, [0,1,2]])
        index = fps(tensor_pcd, ratio=exp_p/len(pcd), random_start=True).numpy()
        pcd = pcd[index]

        return pcd

    def getPointsNumberFromClusters(self, clusters):
        count = 0
        for cluster in clusters:
            count += len(cluster)
        return count

    def getPredict(self, key, feature):
        # 1: obj, 0: env
        path = '/data/nb_models/kitti/gaussian/{}.pkl'.format(key)
        if os.path.exists(path):
            gnb = joblib.load(path)
            y_pred = gnb.predict(feature)
            return y_pred[0]
        else:
            return 1

    def get_imp_env_clusters_coarse(self, points):
        empty_keys = constants.empty_keys
        x_range_list = self.getRangeListFull(points, 0, bin_num = 80)
        y_range_list = self.getRangeListFull(points, 1, bin_num = 60)
        important_point_clusters = []
        env_point_clusters = []
        z_range = self.getMaxRange(points, 2, bin_num = 65)
        for (i, item_x) in enumerate(x_range_list):
            for (j, item_y) in enumerate(y_range_list):
                key = '{}_{}'.format(i, j)
                x_range = item_x[0]
                y_range = item_y[0]
                cluster =  points[((points[:,0] >= x_range[0]) & (points[:,0] <= x_range[1]) & (points[:,1] >= y_range[0]) & (points[:,1] <= y_range[1]))]
                if key not in empty_keys and len(cluster)<3000:                 
                    env_point_clusters.append(cluster)
                else:
                    ground = cluster[(cluster[:,2] <= z_range[1])]
                    imp = cluster[(cluster[:,2] > z_range[1])]
                    important_point_clusters.append(imp)
                    env_point_clusters.append(ground)
        return (important_point_clusters, env_point_clusters)

    def get_imp_env_clusters_fine(self, points):
        empty_keys = self.getEmptyGridKeys()
        x_range_list = self.getRangeListFull(points, 0, bin_num = 80)
        y_range_list = self.getRangeListFull(points, 1, bin_num = 60)
        important_point_clusters = []
        env_point_clusters = []
        z_range = self.getMaxRange(points, 2, bin_num = 65)
        for (i, item_x) in enumerate(x_range_list):
            for (j, item_y) in enumerate(y_range_list):
                key = '{}_{}'.format(i, j)
                x_range = item_x[0]
                y_range = item_y[0]
                cluster =  points[((points[:,0] >= x_range[0]) & (points[:,0] <= x_range[1]) & (points[:,1] >= y_range[0]) & (points[:,1] <= y_range[1]))]
                result = self.getPredict(key, [[item_x[1], item_y[1]]])
                if result == 1:
                    # ground = cluster[(cluster[:,2] <= z_range[1])]
                    # imp = cluster[(cluster[:,2] > z_range[1])]
                    important_point_clusters.append(cluster)
                    # env_point_clusters.append(ground)
                else:
                    env_point_clusters.append(cluster)
        return (important_point_clusters, env_point_clusters)

    def getEnvFilter(self, points, x_range_list, y_range_list, empty_keys):
        env_clusters = []
        for key in empty_keys:
            idx = key.split("_")
            item_x = x_range_list[int(idx[0])]
            item_y = y_range_list[int(idx[1])]
            x_range = item_x[0]
            y_range = item_y[0]
            cluster =  points[((points[:,0] >= x_range[0]) & (points[:,0] <= x_range[1]) & (points[:,1] >= y_range[0]) & (points[:,1] <= y_range[1]))]
            env_clusters.append(cluster)
                
        if len(np.vstack(env_clusters)) < 3000:        
            return True
        else:
            return False

    
    def get_imp_env_clusters_fine2(self, points, pts_filename):
        empty_keys = constants.empty_keys #self.getEmptyGridKeys()
        x_range_list = self.getRangeListFull(points, 0, bin_num = 80)
        y_range_list = self.getRangeListFull(points, 1, bin_num = 60)
        important_point_clusters = []
        env_point_clusters = []
        z_range = self.getMaxRange(points, 2, bin_num = 65)

        isEnvFilterEnable = self.getEnvFilter(points, x_range_list, y_range_list, empty_keys)
        for (i, item_x) in enumerate(x_range_list):
            for (j, item_y) in enumerate(y_range_list):
                key = '{}_{}'.format(i, j)
                x_range = item_x[0]
                y_range = item_y[0]
                cluster =  points[((points[:,0] >= x_range[0]) & (points[:,0] <= x_range[1]) & (points[:,1] >= y_range[0]) & (points[:,1] <= y_range[1]))]
                
                if key not in empty_keys and isEnvFilterEnable:             
                    env_point_clusters.append(cluster)
                else:
                    if key not in empty_keys:
                        # result = self.getPredict(key, [[item_x[1], item_y[1]]])
                        result = constants.nb_dict[pts_filename][key]
                        if result == 1:
                            important_point_clusters.append(cluster)
                    else:
                        if len(cluster) < 10:
                            env_point_clusters.append(cluster)
                        else:
                            ground = cluster[(cluster[:,2] <= z_range[1])]
                            imp = cluster[(cluster[:,2] > z_range[1])]
                            important_point_clusters.append(imp)
                            env_point_clusters.append(ground)
        # np.savez('cluster_{}.npz'.format(constants.counter), *(important_point_clusters))
        return (important_point_clusters, env_point_clusters)

    # def get_imp_env_clusters_efficient(self, points):
    #      total_point_num = points.shape[0]

    #     x_range_list = self.getRangeList(points, 0, False, bin_num = 80)
    #     y_range_list = self.getRangeList(points, 1, False, bin_num = 60)
    #     n_bin = 65

    #     clusters = self.getPointClusters(x_range_list, y_range_list, points, n_bin)


    def statistic_nb(self, points, pts_filename):
        total_point_num = points.shape[0]

        if self.sample_rate > 0.1:
            imp_env = self.get_imp_env_clusters_coarse(points)
        else:
            imp_env = self.get_imp_env_clusters_fine2(points, pts_filename)
        important_point_clusters = imp_env[0]
        env_point_clusters = imp_env[1]

        # print('all: {}'.format(total_point_num))
        # print('num of imp cluster: {}, num of env cluster: {}'.format(len(important_point_clusters), len(env_point_clusters)))
        # print('num of imp points: {}, num of env points: {}'.format(self.getPointsNumberFromClusters(important_point_clusters), self.getPointsNumberFromClusters(env_point_clusters)))
        
        budget = math.ceil(total_point_num * self.sample_rate)
        
        # method 3
        imp_ratio = 0.7
        expected_imp_num = math.ceil(budget * imp_ratio)
        actual_imp_num = self.getPointsNumberFromClusters(important_point_clusters)

        output_points = np.empty((0, points.shape[1]), dtype=np.float32)

        if actual_imp_num == 0:
            imp_ratio = 1
        else:
            imp_ratio = expected_imp_num/actual_imp_num
        if imp_ratio >= 1:
            for cluster in important_point_clusters:
                output_points = np.vstack((output_points,cluster))
        else:
            for cluster in important_point_clusters:
                exp_p = math.ceil(len(cluster) * imp_ratio)

                choice = np.random.choice(cluster.shape[0], exp_p, replace=False)
                output_points = np.vstack((output_points, cluster[choice]))
        # print(len(output_points))
        if len(output_points) > expected_imp_num:
            choice = np.random.choice(output_points.shape[0], expected_imp_num, replace=False)
            output_points = output_points[choice]

            
        expected_env_num = budget - len(output_points)
        # env_ouput = np.empty((0, points.shape[1]), dtype=np.float32)
        
        if expected_env_num > 0:
            env_points = np.vstack((env_point_clusters))
            choice = np.random.choice(env_points.shape[0], expected_env_num, replace=True)
            output_points = np.vstack((output_points, env_points[choice])) 
        return output_points 
        # return np.vstack(important_point_clusters)

    def getEmptyGridKeys(self):
        path = '/home/xiaoyu/experiments/mmdetection3d/tools/nb_models/intersection.json'
        if os.path.exists(path):
            empty_keys = list(json.load(open(path)))
        return empty_keys

    def getRangeListFull(self, points, dim, bin_num):
        count, bins_edge = np.histogram(points[:, dim], bins = bin_num)

        pdfs = count / sum(count)
        # pdfs = np.array(count, dtype=np.int32).tolist()

        output = [] #[[range_start, range_end], pdf]
        for (i, pdf) in enumerate(pdfs):
            range = [bins_edge[i], bins_edge[i+1]]
            record = [range, pdf]
            output.append(record)
        return output

    def getRangeListPeak(self, points, dim, isLarge, bin_num = 200):
        count, bins_edge = np.histogram(points[:, dim], bins = bin_num)

        pdf = count / sum(count)

        median = np.median(pdf)

        start_idx = 0
        end_idx = 0
        for index, num in enumerate(pdf):
            if num >= median:
                start_idx = index
                break
        for index, num in reversed(list(enumerate(pdf))):
            if num >= median:
                end_idx = index
                break

        pdf = pdf[start_idx:(end_idx+1)]
        bins_edge = bins_edge[(start_idx+1):(end_idx+2)]
        ilocs_max = argrelextrema(pdf, np.greater_equal, order=1)[0]


        output = []
        for idx in ilocs_max:
            range = []
            if idx > 1:
                range.append(bins_edge[idx-2])
            else:
                range.append(bins_edge[idx])

            if idx < len(pdf) - 2:
                range.append(bins_edge[idx+2])
            else:
                range.append(bins_edge[idx])

            output.append(range)
        return output

    
    def statistic_sample(self, points, isLarge=False):
        total_point_num = points.shape[0]

        # z_range = self.getMaxRange(points, 2, bin_num = 80)
        # points = points[(points[:,2] > z_range[1]))]

        
        # z_range_list = getRangeList(points, 2, bin_num = 30)
        # if isLarge:
        #     x_range_list = self.getRangeList(points, 0, isLarge, bin_num = 120)
        #     y_range_list = self.getRangeList(points, 1, isLarge, bin_num = 100)
        #     n_bin = 80
        # else:
        x_range_list = self.getRangeList(points, 0, False, bin_num = 120)
        y_range_list = self.getRangeList(points, 1, False, bin_num = 100)
        n_bin = 50

        clusters = self.getPointClusters(x_range_list, y_range_list, points, n_bin)

        budget = math.ceil(total_point_num * self.sample_rate)

        points_important = np.empty((0, points.shape[1]), dtype=np.float32)
        if len(clusters) != 0:
            points_important = np.vstack(clusters)

        # print('total number of points {}, important points {}'.format(total_point_num, points_important.shape[0]))
        
        a_tuples = [tuple(row) for row in points]
        b_tuples = [tuple(row) for row in points_important]

        s = set(b_tuples)

        result_tuples = [x for x in a_tuples if x not in s]

        # result_tuples = np.setdiff1d(a_view, b_view)
        points_env = np.array(result_tuples)
        # print('total {}, important {}, env {}'.format(points.shape[0], points_important.shape[0], points_env.shape[0]))

        output_points = np.empty((0, points.shape[1]), dtype=np.float32)
        
        # method 3
        imp_ratio = 0.7
        expected_imp_num = math.ceil(budget * imp_ratio)

        if len(points_important) != 0:
            imp_ratio = expected_imp_num/len(points_important)
            if imp_ratio >= 1:
                output_points = np.vstack((output_points, points_important))
            else:
                for cluster in clusters:
                    exp_p = math.ceil(len(cluster) * imp_ratio)
                    # lsh_points = self.LSH(cluster, exp_p)
                    # fps_points = self.FPS(cluster, exp_p)
                    # output_points = np.vstack((output_points, lsh_points))

                    choice = np.random.choice(cluster.shape[0], exp_p, replace=False)
                    output_points = np.vstack((output_points, cluster[choice]))
            
        expected_env_num = budget - len(output_points)
        if expected_env_num > 0:
            choice_env = np.random.choice(points_env.shape[0], expected_env_num, replace=False)
            output_points = np.vstack((output_points, points_env[choice_env]))

        # method 1
        
        # imp_ratio = 0.7

        # expected_imp_num = math.ceil(budget * imp_ratio)
        # expected_env_num = math.ceil(budget * (1 - imp_ratio))

        # choice_imp = np.random.choice(points_important.shape[0], expected_imp_num, replace=True)
        # choice_env = np.random.choice(points_env.shape[0], expected_env_num, replace=True)

        # output_points = np.vstack((output_points, points_important[choice_imp]))
        # output_points = np.vstack((output_points, points_env[choice_env]))

        # method 2
        # if budget > points_important.shape[0]:
        #     choice = np.random.choice(points_important.shape[0], math.ceil(points_important.shape[0]*0.8), replace=False)
        #     output_points = np.vstack((output_points, points_important[choice]))
        #     budget_left = budget - output_points.shape[0]

        #     choice = np.random.choice(points_env.shape[0], budget_left, replace=False)
        #     output_points = np.vstack((output_points, points_env[choice]))
        # else:
        #     choice = np.random.choice(points_important.shape[0], budget, replace=False)
        #     output_points = np.vstack((output_points, points_important[choice]))
        
        return output_points

    def getMaxRange(self, points, dim, bin_num = 200):
        count, bins_edge = np.histogram(points[:, dim], bins = bin_num)
        pdf = count / sum(count)
        max_idx = np.argmax(pdf)

        if len(bins_edge) == max_idx+2:
            return [bins_edge[max_idx], bins_edge[max_idx+1]]
        else:
            return [bins_edge[max_idx], bins_edge[max_idx+2]]

    def getPointClusters(self, x_range_list, y_range_list, points, n_bin=50):
        cluster_list = []
        range_list = []
        z_range = self.getMaxRange(points, 2, bin_num = n_bin)

        # z_range = z_range_list[0]
        # print('x range list len {}, y range list len {}'.format(len(x_range_list), len(y_range_list)))
        for x_range in x_range_list:
            for y_range in y_range_list:
                point_cluster = points[((points[:,0] >= x_range[0]) & (points[:,0] <= x_range[1]) & (points[:,1] >= y_range[0]) & (points[:,1] <= y_range[1]) & (points[:,2] > z_range[1]))]
                if len(point_cluster) > 0:
                    range = []
                    cluster_list.append(point_cluster)
                    range.append(x_range)
                    range.append(y_range)
                    range_list.append(range)
        return cluster_list

    def getBinNum(self, points, dim):
        unique, counts = np.unique(points[:, dim], return_counts=True)
        return int(len(unique)/100)



    def getRangeList(self, points, dim, isLarge, bin_num = 200):
        # bin_num = self.getBinNum(points, dim)

        count, bins_edge = np.histogram(points[:, dim], bins = bin_num)

        pdf = count / sum(count)

        if isLarge == False:
        # get range
            median = np.median(pdf)

            start_idx = 0
            end_idx = 0
            for index, num in enumerate(pdf):
                if num >= median:
                    start_idx = index
                    break
            for index, num in reversed(list(enumerate(pdf))):
                if num >= median:
                    end_idx = index
                    break

            pdf = pdf[start_idx:(end_idx+1)]
            bins_edge = bins_edge[(start_idx+1):(end_idx+2)]
            ilocs_max = argrelextrema(pdf, np.greater_equal, order=1)[0]
        else:
            ilocs_max = argrelextrema(pdf, np.greater_equal, order=1)[0]

        output = []
        for idx in ilocs_max:
            range = []
            if idx > 1:
                range.append(bins_edge[idx-2])
            else:
                range.append(bins_edge[idx])

            if idx < len(pdf) - 2:
                range.append(bins_edge[idx+2])
            else:
                range.append(bins_edge[idx])

            output.append(range)
        return output



max_distance = 0
def sim(x, y):
    # print('shape of x: ', x)
    # print('shape of y: ', y)
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return 0
    eu_dist = euclidean_distances(x[:,:3], y[:,:3])[0][0]
    cosine_sim = cosine_similarity(x[:,4:], y[:,4:])
    # cosine_sim = cosine_sim[0][0]
    
    # print('max distance', max_distance)
    # print('eu dist: ', eu_dist)
    # print('cosine dist: ', cosine_sim)

    n_eu_dist = eu_dist/max_distance
    n_cosine_sim = 0.5 * (cosine_sim + 1)

    return 0.5 * (eu_dist/max_distance) + 0.5 * (1 - cosine_sim)

def distance_metric(X):
    # print("shape of X: ", X.shape)
    global max_distance
    max_distance = np.max(euclidean_distances(X[:, :3]))
    return pairwise_distances(X, metric=sim)
