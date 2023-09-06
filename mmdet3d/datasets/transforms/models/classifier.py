import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, k):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(132, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, k)
        # self.softmax = nn.softmax(dim=1)


    def forward(self, x):
        # print('shape of x: ', x.shape)
        x_coordinate = x[:, :, :3]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x_class = torch.argmax(x, dim=2)
        
        return torch.cat((x_coordinate, x_class.unsqueeze(2)), dim=2)

