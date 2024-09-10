import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from model import MyModel, MyModel2
from torch.utils.data import DataLoader
from dataloader import Density_Loader, Density_FPFH_Loader, RawDataset_Loader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
import torch.multiprocessing as mp


def calc_decay(init_lr, epoch):
    return init_lr * 1/(1 + 0.03*epoch)


def train():
    dataset = RawDataset_Loader(split = 'train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)

    test_dataset = RawDataset_Loader(split = 'test')
    testdataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    model = MyModel2(20)
    init_lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)

    model = torch.nn.DataParallel(model)
    model.cuda()

    min_loss = 10000000
    for epoch in range(0, 100):
        model.train()
        lr = calc_decay(init_lr, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        running_loss = 0.0
    
        for data, labels in tqdm(dataloader, total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
            # print(data.shape)
            # print(labels.shape)
            data = data.cuda()
            labels = labels.long().cuda()
            labels = labels.reshape(data.shape[0], 15000, 1)

            logits = model(data)
            #Reshape logits to (4*15000, 2) and labels to (4*15000,)
            logits = logits.view(-1, 2)
            labels = labels.view(-1)

            loss = nn.CrossEntropyLoss()(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
        print('running loss: {}'.format(running_loss))

        if running_loss < min_loss:
            min_loss = running_loss
            fn_pth = 'output/{}_ep{}.pth'.format('point_mlp', epoch)
            torch.save(model.state_dict(), fn_pth)



mp.set_start_method('spawn')
train()