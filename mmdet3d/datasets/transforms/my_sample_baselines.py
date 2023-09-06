from torch_cluster import fps
from torch_cluster import grid_cluster
import open3d as o3d
import numpy as np
import torch

class SampleBaselines():
    def __init__(self, pcd, sample_rate):
        self.pcd = pcd
        self.sample_rate = sample_rate
        self.selected_index = np.empty((0), dtype=np.int32)
        self.oc_node_list = []

    def randomSample(self, pcd):
        length = pcd.shape[0]
        sample_length = int(np.ceil(length * self.sample_rate))
        choice = np.random.choice(length, sample_length, replace=True)
        pcd = pcd[choice]
        
        return pcd
    
    def getRandomIndex(self, pcd):
        length = pcd.shape[0]
        sample_length = int(np.ceil(length * self.sample_rate))
        choice = np.random.choice(length, sample_length, replace=True)

        return choice


    def fpsSample(self, pcd):
        tensor_pcd = torch.from_numpy(pcd[:, [0,1,2]])
        index = fps(tensor_pcd, ratio=self.sample_rate, random_start=True).numpy()

        pcd = pcd[index]

        return pcd
    
    def getFpsIndex(self, pcd):
        tensor_pcd = torch.from_numpy(pcd[:, [0,1,2]])
        index = fps(tensor_pcd, ratio=self.sample_rate, random_start=True).numpy()

        return index

    def gridSample(self, pcd, method='fps'):
        # numpy_pcd = pcd.tensor.detach().cpu().numpy()
        tensor_pcd = torch.from_numpy(pcd[:, [0,1,2]])

        max_coordinates, max_index = torch.max(tensor_pcd, dim=0)
        min_coordinates, min_index = torch.min(tensor_pcd, dim=0)

        diff_x = max_coordinates[0] - min_coordinates[0]
        diff_y = max_coordinates[1] - min_coordinates[1]
        diff_z = max_coordinates[2] - min_coordinates[2]

        # print("x {}, y {}, z {}".format(diff_x, diff_y, diff_z))

        n = 100 # number of grid in each dim
        size = torch.Tensor([diff_x/100, diff_y/100, diff_z/100])

        cluster = grid_cluster(tensor_pcd, size)

        cluster_dict = self.getPointCluster(cluster)

        output_pcd = np.empty((0, pcd.shape[1]), dtype=np.float32)

        for key in cluster_dict:
            indexes = cluster_dict[key]
            
            if method == 'random':
                random_pcd = self.randomSample(pcd[indexes])
                
            else:
                random_pcd = self.fpsSample(pcd[indexes])

            output_pcd = np.append(output_pcd, random_pcd, axis = 0)


        choice = np.random.choice(output_pcd.shape[0], int(pcd.shape[0] * self.sample_rate), replace=True)

        return output_pcd[choice]

    def getPointCluster(self, cluster):
        cluster_dict = {}
        for indx, c in enumerate(cluster.numpy()):
            if c in cluster_dict:
                cluster_dict[c].append(indx)
            else:
                cluster_dict[c] = [indx]
        # print(cluster_dict)
        return cluster_dict
    
    # def getGridIndex(self, pcd):
    #     tensor_pcd = torch.from_numpy(pcd[:, [0,1,2]])

    #     max_coordinates, max_index = torch.max(tensor_pcd, dim=0)
    #     min_coordinates, min_index = torch.min(tensor_pcd, dim=0)

    #     diff_x = max_coordinates[0] - min_coordinates[0]
    #     diff_y = max_coordinates[1] - min_coordinates[1]
    #     diff_z = max_coordinates[2] - min_coordinates[2]

    #     # print("x {}, y {}, z {}".format(diff_x, diff_y, diff_z))

    #     n = 100 # number of grid in each dim
    #     size = torch.Tensor([diff_x/100, diff_y/100, diff_z/100])

    #     cluster = grid_cluster(tensor_pcd, size)

    #     cluster_dict = self.getPointCluster(cluster)

    #     selected = np.empty((0), dtype=np.int32)

    #     for key in cluster_dict:
    #         indexes = cluster_dict[key]
            
    #         index_fps = self.getFpsIndex(pcd[indexes])

    #         indexes_choice = np.asarray(indexes)[index_fps]

    #         selected = np.append(selected, indexes_choice)

    #     # choice = np.random.choice(selected, int(pcd.shape[0] * self.sample_rate), replace=True)

    #     return selected

    # def gridVoxel(self, pcd, grid_size):
    #     tensor_pcd = torch.from_numpy(pcd[:, [0,1,2]])

    #     n = float(grid_size)
    #     size = torch.Tensor([n, n, n])

    #     cluster = grid_cluster(tensor_pcd, size)

    #     cluster_dict = self.getPointCluster(cluster)

    #     # print(len(cluster_dict))

    #     output_pcd = np.empty((0, 4), dtype=np.float32)

    #     for key in cluster_dict:
    #         # if cluster not empty, leave 1 point
    #         # problem is how to control the number of point?
    #         # that is about how to define the parameter grid size.

    #         indexes = cluster_dict[key]
    #         if len(indexes) != 0:
    #             length = pcd.shape[0]
    #             choice = np.random.choice(len(indexes), 1, replace=False)
                
    #             choice_index = indexes[choice[0]]

    #             output_pcd = np.append(output_pcd, [pcd[choice_index]], axis = 0)
            
    #     # print(output_pcd.shape)
    #     # print(output_label.shape)

    #     choice = np.random.choice(output_pcd.shape[0], int(pcd.shape[0] * self.sample_rate), replace=True)

    #     return output_pcd[choice]
    
    # def getGridVoxelIndex(self, pcd, grid_size):
    #     tensor_pcd = torch.from_numpy(pcd[:, [0,1,2]])

    #     n = float(grid_size)
    #     size = torch.Tensor([n, n, n])

    #     cluster = grid_cluster(tensor_pcd, size)

    #     cluster_dict = self.getPointCluster(cluster)

    #     # print(len(cluster_dict))

    #     selected = np.empty((0), dtype=np.int32)

    #     for key in cluster_dict:
    #         indexes = cluster_dict[key]
    #         if len(indexes) != 0:
    #             choice = np.random.choice(len(indexes), 1, replace=False)
    #             choice_index = np.asarray(indexes)[choice[0]]
    #             selected = np.append(selected, choice_index)
            
    #     choice = np.random.choice(selected, int(pcd.shape[0] * self.sample_rate), replace=True)
        
    #     return choice
    
    def octree(self, pcd, tree_depth):
        octree = o3d.geometry.Octree(max_depth=tree_depth)
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])

        octree.convert_from_point_cloud(o3d_pcd, size_expand=0.01)
        # o3d.visualization.draw_geometries([octree])
        octree.traverse(self.f_traverse)

        choice = np.random.choice(self.selected_index.shape[0], int(pcd.shape[0] * self.sample_rate), replace=True)
        self.selected_index = self.selected_index[choice]

        output_pcd = pcd[self.selected_index]

        return output_pcd
    
    def getOctreeIndex(self, pcd, tree_depth):
        octree = o3d.geometry.Octree(max_depth=tree_depth)
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])

        octree.convert_from_point_cloud(o3d_pcd, size_expand=0.01)
        # o3d.visualization.draw_geometries([octree])
        octree.traverse(self.f_traverse)

        # choice = np.random.choice(self.selecte_index.shape[0], int(pcd.shape[0] * self.sample_rate), replace=True)
        # self.selecte_index = self.selecte_index[choice]

        return self.selected_index
    
    def f_traverse(self, node, node_info):
        early_stop = False

        if isinstance(node, o3d.geometry.OctreeInternalNode):
            if isinstance(node, o3d.geometry.OctreeInternalPointNode):
                pass
                # n = 0
                # for child in node.children:
                #     if child is not None:
                #         n += 1
                # print(
                #     "{}{}: Internal node at depth {} has {} children and {} points ({})"
                #     .format('    ' * node_info.depth,
                #             node_info.child_index, node_info.depth, n,
                #             len(node.indices), node_info.origin))

                # we only want to process nodes / spatial regions with enough points
                # early_stop = len(node.indices) < 250
        elif isinstance(node, o3d.geometry.OctreeLeafNode):
            if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
                # print("{}{}: Leaf node at depth {} has {} points with origin {}".
                #     format('    ' * node_info.depth, node_info.child_index,
                #         node_info.depth, len(node.indices), node_info.origin))
                # print(node.indices[:3])
                self.oc_node_list.append(node.indices)
                n_node_selected = int(np.ceil(self.sample_rate * len(node.indices)))
                choice = np.random.choice(len(node.indices), n_node_selected, replace=True)
                # print(np.asarray(node.indices)[choice])
                self.selected_index = np.append(self.selected_index, np.asarray(node.indices)[choice]) 
        else:
            raise NotImplementedError('Node type not recognized!')

        # early stopping: if True, traversal of children of the current node will be skipped
        return early_stop