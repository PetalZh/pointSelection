import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        hidden_size1 = 16
        hidden_size2 = 32
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x))             
        # x = self.sigmoid(x)        
        x = self.fc3(x) 
        x = torch.softmax(x, dim=2)
        return x

class MyModel2(nn.Module):
    def __init__(self, input_size):
        super(MyModel2, self).__init__()
        hidden_size1 = 64
        hidden_size2 = 128
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x))             
        # x = self.sigmoid(x)        
        x = self.fc3(x) 
        x = torch.softmax(x, dim=2)
        return x