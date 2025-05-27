#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 13:43:09 2022

@author: jingrongwang
"""
#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms

#import time

import scipy.io as sio

import numpy as np
#from torchsummary import summary



class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class DataPartitioner_niid(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, indexes, seed=1234):
        self.data = data
        #self.partitions = []
        # rng = Random()
        # rng.seed(seed)
        # data_len = len(data)
        # indexes = [x for x in range(0, data_len)]
        # rng.shuffle(indexes) # random shuffle 0 - (data_len-1)
        self.partitions = indexes
        # for frac in range(len(indexes)):
        #     #part_len = int(frac * data_len)
        #     self.partitions.append(indexes[0:part_len])
        #     indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])



class CNN_Net(nn.Module):
    """ Network architecture. """
      #summary(model,(3,28,28))
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    """On the first case (using dim=1) the softmax function is applied along the axis 1 . That’s why all rows add up to 1. On the second case (using dim=0) the softmax function is applied along the axis 0. Making all the columns add up to 1."""
    """For matrices, it’s 1. For others, it’s 0."""


## deeper model adapted from https://www.kaggle.com/gustafsilva/cnn-digit-recognizer-pytorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # summary(model,(1,28,28))
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 10)
        )
                
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x





class MLP(nn.Module):
    #INPUT_DIM = 28 * 28
    #OUTPUT_DIM = 10
    def __init__(self):
        super().__init__()
                
        self.input_fc = nn.Linear(28 * 28, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, 10)
        
    def forward(self, x):
        
        #x = [batch size, height, width]
        
        batch_size = x.shape[0]

        x = x.view(batch_size, -1)
        
        #x = [batch size, height * width]
        
        h_1 = F.relu(self.input_fc(x))
        
        #h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))

        #h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)
        
        #y_pred = [batch size, output dim]
        
        return y_pred#, h_2



 # Define AlexNet network structure
class AlexNet(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential( # Input 1*28*28
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2), # 32*14*14
            nn.ReLU(inplace=True),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2), # 64*7*7
            nn.ReLU(inplace=True),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128*7*7
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256*7*7
            )
 
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2), # 256*3*3
            nn.ReLU(inplace=True),
            )
        self.fc1 = nn.Linear(256*3*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256*3*3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
 




def partition_dataset():
    """ Partitioning MNIST """
    """ Assuming we have 2 replicas, then each process will have a train_set of 60000 / 2 = 30000 samples. We also divide the batch size by the number of replicas in order to maintain the overall batch size of 128."""
    """CIFAR10, EMNIST,Fashion-MNIST"""
    #print('  start loading dataset')
    #start_time = time.time()
    dataset = datasets.MNIST(
        root = 'data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    
    #print('  dataset',dataset)
    size = dist.get_world_size()
    #print('  size',size)
    bsz = 64#int(256*50/ float(size))#int(256)#int(128 / float(size))
    #print('  bsz',bsz)
    partition_sizes = [1.0 / size for _ in range(size)]
    #print('  partition_sizes',partition_sizes)
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    #print('  train_set',len(train_set))
    return train_set, bsz


def partition_dataset_niid(size, p, bsz):
    """ Partitioning MNIST """

    dataset = datasets.MNIST(
        root = 'data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    
    
    N_class = 10
    
    """ create 10*6000 index list"""
    labels = dataset.targets.tolist()
    index_pos_list = []
    from more_itertools import locate
    for i in range(N_class):
        index_pos = list(locate(labels, lambda a: a == i))
        index_pos_list.append(index_pos)
         
    x = 0
    y = 0
    z = 0
    index_list = []
    N_ = int((len(dataset)/size)/p)
    for i in range(size):
        index_list_worker = []
        for j in range(p):
           index_list_worker =  index_list_worker + index_pos_list[y][z*N_:(z+1)*N_]
           if y == N_class - 1: # move to the next class
               y = 0
               z += 1
           else:
               y += 1
        x += 1 # move to the next worker
        index_list.append(index_list_worker)
    
    partition = DataPartitioner_niid(dataset, index_list)
    train_set = []
    for i in range(size):
        partition_ = partition.use(i)
        train_set_ = torch.utils.data.DataLoader(
        partition_, batch_size=bsz, shuffle=True)
        #print('  train_set',len(train_set))
        train_set.append(train_set_)
    return train_set # n by 60000/n matrixs




def partition_dataset_test():
    """ Partitioning MNIST """
    """ Assuming we have 2 replicas, then each process will have a train_set of 60000 / 2 = 30000 samples. We also divide the batch size by the number of replicas in order to maintain the overall batch size of 128."""
    """CIFAR10, EMNIST,Fashion-MNIST"""
    #print('  start loading dataset')
    #start_time = time.time()
    dataset = datasets.MNIST(
        root = 'data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    
    #print('  dataset',dataset)
    #size = dist.get_world_size()
    # #print('  size',size)
    
    #partition_sizes = [1.0 / size for _ in range(size)]
    # #print('  partition_sizes',partition_sizes)
    
    bsz = 10000#int(128 / float(size))
    # #print('  bsz',bsz)
    
    
    #partition = DataPartitioner(dataset, partition_sizes)
    #partition = partition.use(dist.get_rank())
    test_set = torch.utils.data.DataLoader(
        dataset, batch_size=bsz, shuffle=True)
    #print('  test_set',len(test_set))
    return test_set, bsz


def create_lenet():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, 5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    return model
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 28, 28]             156
              ReLU-2            [-1, 6, 28, 28]               0
         AvgPool2d-3            [-1, 6, 14, 14]               0
            Conv2d-4           [-1, 16, 10, 10]           2,416
              ReLU-5           [-1, 16, 10, 10]               0
         AvgPool2d-6             [-1, 16, 5, 5]               0
           Flatten-7                  [-1, 400]               0
            Linear-8                  [-1, 120]          48,120
              ReLU-9                  [-1, 120]               0
           Linear-10                   [-1, 84]          10,164
             ReLU-11                   [-1, 84]               0
           Linear-12                   [-1, 10]             850
================================================================
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.11
Params size (MB): 0.24
Estimated Total Size (MB): 0.35
----------------------------------------------------------------
"""



def average_gradients(model,W):
    """ Gradient averaging. """
    
    ID = dist.get_rank()
    size = float(dist.get_world_size())
    
    for param in model.parameters():
        #dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)#, group=0)
        #param.grad.data /= size
        
        tensor_shape = param.grad.data.size()
        # receiving Tensor from all ranks
        tensor_list = [
        torch.empty(tensor_shape) for _ in range(int(size))]
        #dist.all_gather(tensor_list, param.grad.data)
        #param.grad.data = sum([ W[ID][i]*tensor_list[i]for i in range(int(size))])
        dist.all_gather(tensor_list, param.  data)
        param.data = sum([ W[ID][i]*tensor_list[i]for i in range(int(size))])
        
        
        #for i in range(len(int(size))):
        #    para = W[ID][i]*tensor_list[i]
            


def run(rank, size):
    """ Distributed Synchronous SGD Example """
    
    #filename = '/Users/jingrongwang/Dropbox/PhD/2_CENT/code/code_0213/dynamic_case1_solutions_50_iter_0302.mat'
    W_type = 5
    range_ = 60
    case = 1
    
    filename = 'CENT_solutions_iter1_'+str(size)+'workers_range'+str(range_)+'.mat'
    
    if W_type == 1: #md
        W = sio.loadmat(filename)['W_md']
    elif W_type == 2:#mh
        W = sio.loadmat(filename)['W_mh']
    elif W_type == 3:#bc
        W = sio.loadmat(filename)['W_bc']
    elif W_type == 4:#fdla
        W = sio.loadmat(filename)['W_fdla']
    elif W_type == 5:#cent
        W = sio.loadmat(filename)['W_cent']
    elif W_type == 6:# fully connected
        W = np.ones((int(size),int(size)))*(1/size) 
    else: # no communication
        W = np.identity(size)
        #W = np.ones((int(size),int(size)))*(1/size)  
        


    n_epoch = 50 
    
    
    torch.manual_seed(10*dist.get_rank())#4321)#1234)
    
    # here i iid data, can also use the non-iid case as in the simulation code
    train_set, bsz = partition_dataset()
    test_set, test_bsz = partition_dataset_test()
    
    """ Train """
    model = create_lenet()#Net() #LeNet()#MLP()#AlexNet()#
    #model = model
    #model = model.cuda(rank)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    cec = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    #num_batches = ceil(len(train_set.dataset) / float(bsz))
    
    
    batch_loss = []
    batch_accuracy = []
    run_time = []
    run_comp_time = []
    
    batch_test_loss = []
    batch_test_accuracy = []
    run_test_time = []
    
    
    iter_count = 0
    for epoch in range(n_epoch):
        
        if dist.get_rank() == 3:
            print('epoch =',epoch)
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_idx = 0
        

        for data, target in train_set:
            batch_idx += 1
        
        
            data, target = Variable(data), Variable(target)
            # data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            #loss = F.nll_loss(output, target)
            loss = cec(output, target)
            
            epoch_loss += loss.item()#data[0]
            loss.backward()
            
            
            average_gradients(model,W)
            
            optimizer.step()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            """append data"""
            batch_loss.append(epoch_loss/(batch_idx+1))
            batch_accuracy.append(100.*correct/total)
            
            if dist.get_rank() == 3:
                if batch_idx %int(len(train_set)/5) == 0:
                    print(batch_idx, len(train_set), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (epoch_loss/(batch_idx+1), 100.*correct/total, correct, total))
            iter_count += 1
            
        if dist.get_rank() == 3:
                """ Test """
                test_epoch_loss = 0.0
                test_correct = 0
                test_total = 0
                test_batch_idx = 0
                
                #test_start_time = time.time()
                
                for data, target in test_set:
                    test_batch_idx += 1
                    
                    data, target = Variable(data), Variable(target)
                    
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    test_epoch_loss += loss.item()#data[0]
                    _, predicted = output.max(1)
                    test_total += target.size(0)
                    test_correct += predicted.eq(target).sum().item()
                    print(test_batch_idx, len(test_set), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_epoch_loss/(test_batch_idx+1), 100.*test_correct/test_total, test_correct, test_total))

                """append data"""
                batch_test_loss.append(test_epoch_loss/(test_batch_idx+1))
                batch_test_accuracy.append(100.*test_correct/test_total)


    if dist.get_rank() == 3:
        mdic = {'batch_training_loss':batch_loss,
                'batch_training_accuracy':batch_accuracy,
                'run_training_time':run_time,
                'run_comp_time':run_comp_time,
                'batch_test_loss':batch_test_loss,
                'batch_test_accuracy':batch_test_accuracy,
                'run_test_time':run_test_time,
                'n_epoch':n_epoch,
                'worker_ID':dist.get_rank()}


        if W_type == 1:
            sio.savemat("worker_"+str(dist.get_rank())+"_of_"+str(size)+'_'+str(n_epoch)+"_epochs_Max-degree_range"+str(range_)+str(case)+".mat", mdic)
        elif W_type == 2:
            sio.savemat("worker_"+str(dist.get_rank())+"_of_"+str(size)+'_'+str(n_epoch)+"_epochs_Metropolis_range"+str(range_)+str(case)+".mat", mdic)
        elif W_type == 3:
            sio.savemat("worker_"+str(dist.get_rank())+"_of_"+str(size)+'_'+str(n_epoch)+"_epochs_Best-constant_range"+str(range_)+str(case)+".mat", mdic)
        elif W_type == 4:
            sio.savemat("worker_"+str(dist.get_rank())+"_of_"+str(size)+'_'+str(n_epoch)+"_epochs_FDLA_range"+str(range_)+str(case)+".mat", mdic)
        elif W_type == 5:
            sio.savemat("worker_"+str(dist.get_rank())+"_of_"+str(size)+'_'+str(n_epoch)+"_epochs_CENT_range"+str(range_)+str(case)+"_static.mat", mdic)
        elif W_type == 6:
            sio.savemat("worker_"+str(dist.get_rank())+"_of_"+str(size)+'_'+str(n_epoch)+"_epochs_Fully-connected_range"+str(range_)+str(case)+".mat", mdic)
        else:
            sio.savemat("worker_"+str(dist.get_rank())+"_of_"+str(size)+'_'+str(n_epoch)+"_epochs_No-consensus.mat", mdic)
            
        
        

def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    
   
    size = 8
    processes = []
    for rank in range(size):
        print('rank = ', rank)
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
