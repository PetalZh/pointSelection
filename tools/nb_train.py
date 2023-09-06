from mmdet3d.datasets.transforms.kitti_utils import kitti_util
from mmdet3d.datasets.transforms.kitti_utils.kitti_util import Calibration
from numpy import genfromtxt
import numpy as np
import os
import json
from sklearn.naive_bayes import GaussianNB, CategoricalNB, ComplementNB
import joblib

def getEmptyCubeKeys():
    parts = [1000, 2000, 3000, 4000, 5000, 6000, 7480]
    intersect = getFullKeys()
    for part in parts:
        path = '/home/xiaoyu/experiments/mmdetection3d/tools/output/bays_{}.json'.format(part)
        print(path)
        list_no_data = loadData(path)
        intersect = list(set(intersect) & set(list_no_data))
    write_to_file(intersect)

def train():
    parts = [1000, 2000, 3000, 4000, 5000, 6000, 7000] #7480
    for i in range(0, 80):
        for j in range(0, 60):
            key = '{}_{}'.format(i, j)
            features = []
            labels = [] # 1: obj, 0: env
            getData(parts, key, features, labels) #training
            print('fitting model for {}'.format(key))
            # gnb = GaussianNB()
            # gnb.fit(features, labels)
            # joblib.dump(gnb, '/data/nb_models/kitti/{}.pkl'.format(key))
            compNB = ComplementNB(force_alpha=True)
            compNB.fit(features, labels)
            joblib.dump(compNB, '/data/nb_models/kitti/complement/{}.pkl'.format(key))

            cateNB = CategoricalNB(force_alpha=True)
            cateNB.fit(features, labels)
            joblib.dump(cateNB, '/data/nb_models/kitti/categorical/{}.pkl'.format(key))


            
def test():
    parts = [7480] #7480
    miss_num = []
    for i in range(0, 28):
            for j in range(0, 60):
                key = '{}_{}'.format(i, j)
                print(key)

                features = []
                labels = [] # 1: obj, 0: env
                getData(parts, key, features, labels) #training

                path = '/data/nb_models/kitti/{}.pkl'.format(key)
                gnb = joblib.load(path)
                
                y_pred = gnb.predict(features)
                # print("Number of mislabeled points out of a total %d points : %d" % (len(features), (labels != y_pred).sum()))
                # total_num.append(len(features))
                miss_num.append((labels != y_pred).sum())
    print(miss_num)
    print('missing percentage: {}'.format(sum(miss_num)/(480*len(miss_num))))


def getData(parts, key, features, labels):
    for part in parts:
        path = '/home/xiaoyu/experiments/mmdetection3d/tools/output/bays_{}.json'.format(part)
        print(path)
        sta_dict = json.load(open(path))
        content = sta_dict[key] # 'obj_x', 'obj_y', 'env_x', 'env_y'

        for idx in range(0, len(content['obj_x'])):
            features.append([content['obj_x'][idx], content['obj_y'][idx]])
            labels.append(1)
        for idx in range(0, len(content['env_x'])):
            features.append([content['env_x'][idx], content['env_y'][idx]])
            labels.append(0)
            

def getObjCount():
    parts = [1000, 2000, 3000, 4000, 5000, 6000]
    obj_dict = {}
    for part in parts:
        path = '/home/xiaoyu/experiments/mmdetection3d/tools/output/bays_{}.json'.format(part)
        print(path)
        
        createObjDict(path, obj_dict)
    
    sorted_dict = sorted(obj_dict.items(),  key=lambda item: item[1])
    for key in sorted_dict[:10]:
        print('key {}, value {}'.format(key, obj_dict[key]))
        
def createObjDict(path, obj_dict):
    if os.path.exists(path):
        sta_dict = json.load(open(path))
        for i in range(0, 80):
            for j in range(0, 60):
                key = '{}_{}'.format(i, j)
                if len(sta_dict[key]['obj_x']) != 0:
                    # print(sta_dict[key]['obj_x'])
                    if key in obj_dict.keys():
                        obj_dict[key] += len(sta_dict[key]['obj_x'])
                    else:
                        obj_dict[key] = len(sta_dict[key]['obj_x'])
                # print(sta_dict[key].keys())
                # print(len(sta_dict[key]['obj_x']))
                # print(len(sta_dict[key]['obj_y']))
                # print(len(sta_dict[key]['env_x']))
                # print(len(sta_dict[key]['env_y']))

    
def write_to_file(intersect):
    output_path = '/home/xiaoyu/experiments/mmdetection3d/tools/nb_models/intersection.json'
    if not os.path.exists(output_path):
            f = open(output_path, "x")
    json.dump(intersect, open(output_path, 'w'))

def getFullKeys():
    keys = []
    for i in range(0, 80):
        for j in range(0, 60):
            key = '{}_{}'.format(i, j)
            keys.append(key)
    return keys

def loadData(path):
    output_list = []
    if os.path.exists(path):
        sta_dict = json.load(open(path))
        for i in range(0, 80):
            for j in range(0, 60):
                key = '{}_{}'.format(i, j)
                if len(sta_dict[key]['obj_x']) == 0:
                    output_list.append(key)
                # print(sta_dict[key].keys())
                # print(len(sta_dict[key]['obj_x']))
                # print(len(sta_dict[key]['obj_y']))
                # print(len(sta_dict[key]['env_x']))
                # print(len(sta_dict[key]['env_y']))
    return output_list

train()
# test()
# getObjCount()