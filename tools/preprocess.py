from numpy import genfromtxt
import numpy as np
import os
import json
import joblib

# def getPCPath():
#     file_path = 'pc_list.json'
#     with open(file_path, 'r') as f:
#         path_list = json.load(f)

#     output_dict = {}
#     for (i, path) in enumerate(path_list):
#         processPC(path, output_dict)
#         if i % 100 == 0:
#             print(str(i))

#     path = '/home/xiaoyu/experiments/mmdetection3d/tools/pc_preprocess.json'
#     if not os.path.exists(path):
#         f = open(path, "x")
#     json.dump(output_dict, open(path, 'w'))

def preprocess():
    file_path = 'pc_list.json'
    with open(file_path, 'r') as f:
        path_list = json.load(f)
    pc_dict = createPCDict(path_list)
    result_dict = getResultDict(path_list, pc_dict)

    path = '/home/xiaoyu/experiments/mmdetection3d/tools/pc_preprocess_complement.json'
    if not os.path.exists(path):
        f = open(path, "x")
    json.dump(result_dict, open(path, 'w'))

def getResultDict(path_list, pc_dict):
    result_dict = {}
    for i in range(80):
        for j in range(60):
            key = '{}_{}'.format(i, j)
            feature_list = []
            for path in path_list:
                item_x = pc_dict[path][0][i]
                item_y = pc_dict[path][1][j]
                feature_list.append([item_x[1], item_y[1]])
            result_list = getPredictBatch(key, feature_list)
            
            for path, result in zip(path_list, result_list):
                if path not in result_dict.keys():
                    result_dict[path] = {key: str(result)}
                else:
                    result_dict[path][key] = str(result)
        if i % 8 == 0:
            print(str(i))
    return result_dict



def createPCDict(path_list):
    pc_dict = {}
    for path in path_list:
        points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1,4])
        x_range_list = getRangeListFull(points, 0, bin_num = 80)
        y_range_list = getRangeListFull(points, 1, bin_num = 60)

        if path not in pc_dict.keys():
            pc_dict[path] = [x_range_list, y_range_list]
    return pc_dict


def getRangeListFull(points, dim, bin_num):
    count, bins_edge = np.histogram(points[:, dim], bins = bin_num)

    pdfs = count / sum(count)
    # pdfs = np.array(count, dtype=np.int32).tolist()

    output = [] #[[range_start, range_end], pdf]
    for (i, pdf) in enumerate(pdfs):
        range = [bins_edge[i], bins_edge[i+1]]
        record = [range, pdf]
        output.append(record)
    return output

def processPC(path, output_dict):
    points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1,4])
    x_range_list = getRangeListFull(points, 0, bin_num = 80)
    y_range_list = getRangeListFull(points, 1, bin_num = 60)

    for (i, item_x) in enumerate(x_range_list):
        for (j, item_y) in enumerate(y_range_list):
            key = '{}_{}'.format(i, j)
            x_range = item_x[0]
            y_range = item_y[0]
            result = getPredict(key, [[item_x[1], item_y[1]]])

            if path not in output_dict.keys():
                output_dict[path] = {key:str(result)}
            else:
                output_dict[path][key] = str(result)

def getPredictBatch(key, feature):
    # 1: obj, 0: env
    path = '/data/nb_models/kitti/categorical/{}.pkl'.format(key)
    if os.path.exists(path):
        gnb = joblib.load(path)
        y_pred = gnb.predict(feature)
        return y_pred
    else:
        return []


def getPredict(key, feature):
    # 1: obj, 0: env
    path = '/data/nb_models/kitti/gaussian/{}.pkl'.format(key)
    if os.path.exists(path):
        gnb = joblib.load(path)
        y_pred = gnb.predict(feature)
        return y_pred[0]
    else:
        return 1




# getPCPath()
preprocess()