import torch
from torch.utils.data import Dataset
import json
from models.pointnet import PointNetSeg
from kitti_utils import kitti_util
from kitti_utils.kitti_util import Calibration
import numpy as np
import os


class Density_Loader(Dataset):
    def __init__(self, split = 'train'):
        # self.data,  self.labels = self.getData()
        self.empty_keys = self.getEmptyGridKeys()
        self.mapping  = self.getKeyMapping()
        
        if split == 'train':
            self.data,  self.labels = self.getAllData( [1000, 2000, 3000, 4000, 5000, 6000, 7000]) #, 2000, 3000, 4000, 5000, 6000, 7000
        else:
            self.data,  self.labels = self.getAllData([7480])#self.getAllData( [7480])#self.getData()#self.getAllData( [7480])#self.getTest()


        print(len(self.data[0]))

    def getKeyMapping(self):
        mapping = {}
        idx = 0
        for i in range(0, 80):
            for j in range(0, 60):
                key = '{}_{}'.format(i, j)
                mapping[key] = idx
                idx += 1
        return mapping


    def getEmptyGridKeys(self):
        path = '/home/xiaoyu/experiments/mmdetection3d/tools/nb_models/intersection_5.json'
        if os.path.exists(path):
            empty_keys = list(json.load(open(path)))
        print('len of empty: {}'.format(len(empty_keys)))
        return empty_keys

    def getIndice(self):
        indice = []
        for key in self.empty_keys:
            indice.append(self.mapping[key])
        return indice

    def getAllData(self, parts):
        # parts = [1000] #7480, 2000, 3000, 4000, 5000, 6000, 7000
        data = []
        labels = []

        for part in parts:
            path = '/home/xiaoyu/experiments/mmdetection3d/tools/output/mlp_{}.json'.format(part)
            print(path)
            sta_dict = json.load(open(path))

            keys = sorted(sta_dict.keys())
            empty_indices = self.getIndice()

            for key in keys:
                item_data = sta_dict[key]['data']
                item_labels = sta_dict[key]['labels']

                item_data = [item_data[i] for i in range(len(item_data)) if i not in empty_indices]
                item_labels = [item_labels[i] for i in range(len(item_labels)) if i not in empty_indices]
                data.append(item_data)
                labels.append(item_labels)
        
        return data, labels

    def getTest(self):
        data = []
        labels = []
        path = '/home/xiaoyu/experiments/mmdetection3d/tools/output/mlp_{}.json'.format(7480)
        print(path)
        sta_dict = json.load(open(path))

        for key in sta_dict.keys(): #7481
            data.append(sta_dict[key]['data'])
            labels.append(sta_dict[key]['labels'])
        return data, labels


    def getData(self):
        parts = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 7480] #7480
        data = []
        labels = []
        path = '/home/xiaoyu/experiments/mmdetection3d/tools/output/mlp_{}.json'.format(1000)
        print(path)
        sta_dict = json.load(open(path))

        for i in range(1, 1001): #7481
            key = f"{i:06}"
            data.append(sta_dict[key]['data'])
            labels.append(sta_dict[key]['labels'])
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_tensor = torch.tensor(self.data[index], dtype=torch.float)
        label_tensor = torch.tensor(self.labels[index], dtype=torch.int)
        return data_tensor, label_tensor



class Density_FPFH_Loader(Dataset):
    def __init__(self, split = 'train'):
        # self.data,  self.labels = self.getData()
        self.empty_keys = self.getEmptyGridKeys()
        self.mapping  = self.getKeyMapping()
        if split == 'train':
            self.data,  self.labels = self.getAllData() #self.getData()
        else:
            self.data,  self.labels = self.getTest()

    def getKeyMapping(self):
        mapping = {}
        idx = 0
        for i in range(0, 80):
            for j in range(0, 60):
                key = '{}_{}'.format(i, j)
                mapping[key] = idx
                idx += 1
        return mapping


    def getEmptyGridKeys(self):
        path = '/home/xiaoyu/experiments/mmdetection3d/tools/nb_models/intersection_5.json'
        if os.path.exists(path):
            empty_keys = list(json.load(open(path)))
        print('len of empty: {}'.format(len(empty_keys)))
        return empty_keys

    def getIndice(self):
        indice = []
        for key in self.empty_keys:
            indice.append(self.mapping[key])
        return indice

    def getAllData(self):
        parts = [1000, 2000, 3000, 4000, 5000, 6000, 7000] #7480
        data = []
        labels = []
        empty_indices = self.getIndice()

        for part in parts:
            path = '/home/xiaoyu/experiments/mmdetection3d/tools/output/mlp_FPFH{}.json'.format(part)
            print(path)
            sta_dict = json.load(open(path))

            keys = sorted(sta_dict.keys())
            
            for key in sta_dict.keys():
                FPFH_data = self.processFPFHData(sta_dict[key]['data'])
                item_labels = sta_dict[key]['labels']

                item_data = [FPFH_data[i] for i in range(len(FPFH_data)) if i not in empty_indices]
                item_labels = [item_labels[i] for i in range(len(item_labels)) if i not in empty_indices]

                
                data.append(item_data)
                labels.append(item_labels)
        return data, labels

    def getTest(self):
        data = []
        labels = []
        path = '/home/xiaoyu/experiments/mmdetection3d/tools/output/mlp_FPFH{}.json'.format(7480)
        print(path)
        sta_dict = json.load(open(path))

        for key in sta_dict.keys(): #7481
            data.append(sta_dict[key]['data'])
            labels.append(sta_dict[key]['labels'])
        return data, labels


    def getData(self):
        parts = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 7480] #7480
        data = []
        labels = []
        path = '/home/xiaoyu/experiments/mmdetection3d/tools/output/mlp_FPFH{}.json'.format(1000)
        print(path)
        sta_dict = json.load(open(path))

        for i in range(1, 1001): #7481
            key = f"{i:06}"
            FPFH_data = self.processFPFHData(sta_dict[key]['data'])
            data.append(FPFH_data)
            labels.append(sta_dict[key]['labels'])
        return data, labels

    def processFPFHData(self, data):
        new_data = []
        for item in data:
            item[2].insert(0, item[1])
            item[2].insert(0, item[0])
            new_data.append(item[2])
        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_tensor = torch.tensor(self.data[index], dtype=torch.float)
        label_tensor = torch.tensor(self.labels[index], dtype=torch.int)
        return data_tensor, label_tensor


class RawDataset_Loader(Dataset):
    def __init__(self, split = 'train'):

        self.labels = self.loadLabel()
        self.length = len(self.labels.keys())

    def loadLabel(self):
        parts = [1000, 2000, 3000, 4000, 5000, 6000] #, 2000, 3000, 4000, 7480 , 5000, 6000, 7000
        output = {}

        for part in parts:
            path = 'labels{}.json'.format(part)
            print(path)
            label_dict = json.load(open(path))
            output.update(label_dict)
        return output

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

    def getData(self, index):
        formatted_number = f"{index:06}"
        path = '/data/kitti_detection_3d/training/velodyne_reduced/{}.bin'.format(formatted_number)
        label_path = '/data/kitti_detection_3d/training/label_2/{}.txt'.format(formatted_number)

        points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1,4])

        sample_length = 15000
        choice = np.random.choice(points.shape[0], sample_length, replace=True)
        points = points[choice]

        feature = self.getFeature(points)
        labels = np.array(self.getLabels(index))[choice]

        return feature, labels

    def getLabels(self, index):
        formatted_number = f"{index:06}"
        return self.labels[formatted_number]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data, label = self.getData(index+1)
        # label_tensor = torch.tensor(self.labels[index], dtype=torch.int)
        return data, label