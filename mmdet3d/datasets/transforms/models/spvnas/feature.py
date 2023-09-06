from .model_zoo import spvnas, spvcnn
import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
import numpy as np

def spvnarsFeature(points, name = 'SemanticKITTI_val_SPVNAS@65GMACs'):
    # points = torch.from_numpy(points[:, :4]).float().cuda()
    block_ = points.astype(np.float32).reshape(-1, 4)
    block = np.zeros_like(block_)
    
    theta = 0.0
    transform_mat = np.array([[np.cos(theta),
                                np.sin(theta), 0],
                                [-np.sin(theta),
                                np.cos(theta), 0], [0, 0, 1]])

    block[...] = block_[...]
    block[:, :3] = np.dot(block[:, :3], transform_mat)
    block[:, 3] = block_[:, 3]

    pc_ = np.round(block[:, :4] / 0.05).astype(np.int32)
    pc_ -= pc_.min(0, keepdims=1)

    # print('shape of block: ', block.shape)
    # print('shape of pc: ', pc_.shape)

    feat_ = block

    # _, inds, inverse_map = sparse_quantize(pc_,
    #                                         return_index=True,
    #                                         return_inverse=True)
                                            
    # pc = torch.from_numpy(pc_[inds]).cuda()
    # feat = torch.from_numpy(feat_[inds]).cuda()

    pc = torch.from_numpy(pc_).cuda()
    feat = torch.from_numpy(feat_).cuda()

    lidar = SparseTensor(feat, pc)

    model = spvnas(name)
    with torch.no_grad():
        model.eval()
        output, intermediate = model(lidar)
    
    return intermediate[0]


def spvcnnFeature(points, name = 'SemanticKITTI_val_SPVCNN@119GMACs'):
    # points = torch.from_numpy(points[:, :4]).float().cuda()
    block_ = points.astype(np.float32).reshape(-1, 4)
    block = np.zeros_like(block_)
    
    theta = 0.0
    transform_mat = np.array([[np.cos(theta),
                                np.sin(theta), 0],
                                [-np.sin(theta),
                                np.cos(theta), 0], [0, 0, 1]])

    block[...] = block_[...]
    block[:, :3] = np.dot(block[:, :3], transform_mat)
    block[:, 3] = block_[:, 3]

    pc_ = np.round(block[:, :4] / 0.05).astype(np.int32)
    pc_ -= pc_.min(0, keepdims=1)

    feat_ = block

    pc = torch.from_numpy(pc_).cuda()
    feat = torch.from_numpy(feat_).cuda()

    lidar = SparseTensor(feat, pc)

    model = spvcnn(name)
    with torch.no_grad():
        model.eval()
        output, intermediate = model(lidar)
    
    return intermediate[0]