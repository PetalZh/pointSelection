import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from model import MyModel
from torch.utils.data import DataLoader
from dataloader import Density_Loader, Density_FPFH_Loader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

# def getData():
#     parts = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 7480] #7480
#     path = '/home/xiaoyu/experiments/mmdetection3d/tools/output/mlp_{}.json'.format(1000)
#     print(path)
#     sta_dict = json.load(open(path))

#     for i in range(1, 1001): #7481
#         key = f"{i:06}"
#         print(key)
#     # return (data, labels)

# getData()

def calc_decay(init_lr, epoch):
    return init_lr * 1/(1 + 0.03*epoch)

def eval_density(model, dataloader):
    accuracy = []
    precision_list = []
    recall_list = []
    for data, labels in dataloader:#tqdm(dataloader, total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
        data = data.cuda()
        labels = labels.long().cuda()
        labels = labels.reshape(data.shape[0], data.shape[1], 1)
        with torch.no_grad():
            pred = model(data)

            pred = pred.argmax(-1)
            # print(labels[:2])
            
            # # reshape
            # logits = logits.view(-1, 2)
            pred = pred.view(-1)
            labels = labels.view(-1)

            correct = (pred == labels).sum().cpu().item()

            # print(correct)

            acc_score = correct/ (4 * 4800)
            print('acc: {}'.format(acc_score))
            
            accuracy.append(acc_score) #(batch_size * num_region)

            # labels = labels.cpu()
            # pred = pred.cpu()

            pred = pred.view(-1).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()

            precision = precision_score(labels, pred)
            print('precision: {}'.format(precision))
            # recall = recall_score(labels, pred)
            if precision != 0:
                precision_list.append(precision)

            # print('precision: {}, acc: {}'.format(precision, acc_score))
            # print(precision)
            # print(recall)

            # precision_list.append(precision)
            # recall_list.append(recall)

            # print(correct)
    if len(precision_list) == 0:
        precision = 0
    else:
        precision = np.mean(precision_list)
    # print(accuracy)
        # print(precision)
    # precision = np.mean(precision_list)
    # recall = np.mean(recall_list)
    # print(acc)
    # print(precision)
    # print(recall)
    return precision



def train_density():
    dataset = Density_Loader(split = 'train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = Density_Loader(split = 'test')
    testdataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    model = MyModel(2)
    init_lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)

    model = torch.nn.DataParallel(model)
    model.cuda()
    max_precision = 0
    for epoch in range(0, 100):
        model.train()
        lr = calc_decay(init_lr, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        running_loss = 0.0
        for data, lables in tqdm(dataloader, total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
            data = data.cuda()
            labels = lables.long().cuda()
            labels = labels.reshape(data.shape[0], data.shape[1], 1)

            logits = model(data)
            # print(logits[:3])
            # print(logits.shape)
            # print(labels.shape)

            # Reshape logits to (4*4800, 2) and labels to (4*4800,)
            logits = logits.view(-1, 2)
            labels = labels.view(-1)

            loss = nn.CrossEntropyLoss()(logits, labels)

            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
        print('running loss: {}'.format(running_loss))

        fn_pth = 'output/density_mlp_opt.pth'
        torch.save(model.state_dict(), fn_pth)

        # precision = eval_density(model, testdataloader)

        # if precision > max_precision:
        #     max_precision = precision
        #     fn_pth = 'output/density_mlp_opt{}.pth'.format('epoch')
        #     torch.save(model.state_dict(), fn_pth)

        # eval(model, testdataloader)
        # fn_pth = '{}.pth'.format('density_mlp')


def train_density_FPFH():
    dataset = Density_FPFH_Loader(split = 'train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    # test_dataset = Density_FPFH_Loader(split = 'test')
    # testdataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    model = MyModel(10)
    init_lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)

    model = torch.nn.DataParallel(model)
    model.cuda()

    for epoch in range(0, 100):
        model.train()
        lr = calc_decay(init_lr, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        running_loss = 0.0
        for data, lables in tqdm(dataloader, total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
            data = data.cuda()
            labels = lables.long().cuda()
            labels = labels.reshape(data.shape[0], data.shape[1], 1)

            logits = model(data)
            # print(logits[:3])
            # print(logits.shape)
            # print(labels.shape)

            # Reshape logits to (4*4800, 2) and labels to (4*4800,)
            logits = logits.view(-1, 2)
            labels = labels.view(-1)

            loss = nn.CrossEntropyLoss()(logits, labels)

            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
        print('running loss: {}'.format(running_loss))

        # eval(model, testdataloader)
        fn_pth = 'output/{}.pth'.format('density_FPFH_mlp_opt')
        torch.save(model.state_dict(), fn_pth)

def test_density():
    test_dataset = Density_Loader(split = 'test')
    testdataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    model = MyModel(2)
    model = torch.nn.DataParallel(model)
    # path = 'density_mlp.pth'
    path = 'output/density_mlp_opt.pth'#'density_mlp.pth'#'output/density_mlp_opt.pth'
    state_dict = torch.load(path)
    model.cuda()

    eval_density(model, testdataloader)

    # print(model.state_dict().keys())
    # print(state_dict.keys())

    # model.load_state_dict(state_dict)
    # for data, labels in tqdm(testdataloader, total=len(testdataloader), smoothing=0.9, dynamic_ncols=True):
    #     with torch.no_grad():
    #         data = data.cuda()
    #         pred = model(data)
    #         pred = pred.argmax(-1)
    #         pred = pred.view(-1)

    #         print(pred)



# train_density()
test_density()

# train_density_FPFH()
