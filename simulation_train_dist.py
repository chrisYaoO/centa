#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:06:09 2022

@author: jingrongwang
"""
#import os
import torch
#import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from math import ceil
from random import Random
#from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
import argparse
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
        rng.shuffle(indexes) # random shuffle 0 - (data_len-1)

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

    dataset = datasets.MNIST(
        root = 'data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    bsz = 10000

    test_set = torch.utils.data.DataLoader(
        dataset, batch_size=bsz, shuffle=True)

    return test_set


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



def communications(model,W):
    """ Gradient averaging. """

    
    N_parameters = 0#len(model[0].parameters())
    
    for param in model[0].parameters():
        #print('param',param)
        #print('param',type(param))
        N_parameters += 1
    # print('    each model has',N_parameters,'parameters\n')
        
    size = len(model)
    tensor_list = []
    for n in range(N_parameters):
        tensor_list_ = []
        for i in range(size):
            idx = 0
            for param in model[i].parameters():
                if idx == n:
                    # if i == 0:
                    #     print(param.data)
                    tensor_list_.append(param.data)
                idx += 1
        tensor_list.append(tensor_list_)

    for i in range(len(model)):
        idx = 0
        for param in model[i].parameters():
            param.data = sum([ W[i][j]*tensor_list[idx][j]for j in range(int(size))])
            idx += 1
            # if i == 0:
            #     print('local model param.data', param.data )
            #     print('length', len(param.data))
        

def aggragate_global_model(model, global_model):  
    N_parameters = 0#len(model[0].parameters())
    
    for param in model[0].parameters():
        #print('param',param)
        #print('param',type(param))
        N_parameters += 1
        
    size = len(model)
    tensor_list = []
    for n in range(N_parameters):
        tensor_list_ = []
        for i in range(size):
            idx = 0
            for param in model[i].parameters():
                if idx == n:
                    tensor_list_.append(param.data)
                idx += 1
        tensor_list.append(tensor_list_)
    idx = 0
    for param in global_model.parameters():
        param.data = sum([ (1/size)*tensor_list[idx][j]for j in range(int(size))])
        idx += 1
        # print('global model param.data', param.data )
        # print('length', len(param.data))

      

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Specify the variables of distributed learning')
    parser.add_argument('-n','--n', help='Number of workers' , type=int, default=8)
    parser.add_argument('-p','--p', help='Number of classes per worker', type=int, default=10)
    parser.add_argument('-r','--range_', help='Communication range', type=int, default=60)
    parser.add_argument('-w','--W_type', help='Work type: 1md, 2mh, 3bc, 4fdla, 5cent, 6fully, 7identity', type=int, default=6)
    parser.add_argument('-b','--bsz', help='minibatch size', type=int, default=512)
    parser.add_argument('-e','--epoch', help='Number of epochs', type=int, default=150)
    
    args = parser.parse_args()
    size = args.n
    p = args.p
    range_ = args.range_
    W_type = args.W_type
    bsz = args.bsz
    n_epoch = args.epoch
    

    
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
    
    
    train_set = partition_dataset_niid(size,p, bsz)
    test_set = partition_dataset_test()
    
    model = []
    optimizer = []

    
    batch_loss = []
    batch_accuracy = []
    run_time = []
    run_comp_time = []
    
    batch_test_loss = []
    batch_test_accuracy = []
    run_test_time = []
    
    #print('creating models ... \n')
    for i in range(size):
        model_ = create_lenet()#Net() #LeNet()#MLP()#AlexNet()#
        optimizer_ = optim.Adam(model_.parameters(), lr=1e-3)
        cec = nn.CrossEntropyLoss()
        model.append(model_)
        optimizer.append(optimizer_)
    #print('create',len(model),'models\n')
    
    global_model = create_lenet()
    #print('create a global model\n')
    
    #print('start training\n')
    for epoch in range(n_epoch):
        #print('Epoch =',epoch)
        for i in range(size):
            #print('    worker',i,'is')
            batch_idx = 0
            for data, target in train_set[i]:
                batch_idx += 1
                data, target = Variable(data), Variable(target)
                optimizer[i].zero_grad()
                output = model[i](data)
                #loss = F.nll_loss(output, target)
                loss = cec(output, target)
                
                #epoch_loss += loss.item()#data[0]
                loss.backward()
                optimizer[i].step()
                
        # communication
        #print('    start communication...')    
        communications(model,W) 
        
        # test accuracy
        #print('    aggragating the global model ... ')
        aggragate_global_model(model, global_model)
        
        # for param in global_model.parameters():
        #     print('global model param.data', param.data )
        #     print('length', len(param.data))
        
        """ Test """
        #print('    start testing ... ')
        test_epoch_loss = 0.0
        test_correct = 0
        test_total = 0
        test_batch_idx = 0
        
        #test_start_time = time.time()
        
        for data, target in test_set:
            test_batch_idx += 1
            
            data, target = Variable(data), Variable(target)
            
            output = global_model(data)
            loss = F.nll_loss(output, target)
            test_epoch_loss += loss.item()#data[0]
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
            #if test_batch_idx == 1:
            print('Epoch =',epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (test_epoch_loss/(test_batch_idx+1), 100.*test_correct/test_total, test_correct, test_total))
        
        
        #test_time_iter = time.time() - test_start_time
        """append data"""
        batch_test_loss.append(test_epoch_loss/(test_batch_idx+1))
        batch_test_accuracy.append(100.*test_correct/test_total)  
        
        if epoch %5 == 0:
            mdic = {'batch_training_loss':batch_loss,
                    'batch_training_accuracy':batch_accuracy,
                    'run_training_time':run_time,
                    'run_comp_time':run_comp_time,
                    'batch_test_loss':batch_test_loss,
                    'batch_test_accuracy':batch_test_accuracy,
                    'run_test_time':run_test_time,
                    'n_epoch':n_epoch} 
            date = "0314"
            if W_type == 1:
                sio.savemat(date+'_'+str(size)+'_workers_'+str(n_epoch)+'_epochs_'+str(p)+'_class_range'+str(range_)+'_Max-degree.mat', mdic)
            elif W_type == 2:
                sio.savemat(date+'_'+str(size)+'_workers_'+str(n_epoch)+'_epochs_'+str(p)+'_class_range'+str(range_)+'_Metropolis.mat', mdic)
            elif W_type == 3:
                sio.savemat(date+'_'+str(size)+'_workers_'+str(n_epoch)+'_epochs_'+str(p)+'_class_range'+str(range_)+'_Best-constant.mat', mdic)
            elif W_type == 4:
                sio.savemat(date+'_'+str(size)+'_workers_'+str(n_epoch)+'_epochs_'+str(p)+'_class_range'+str(range_)+'_FDLA.mat', mdic)
            elif W_type == 5:
                sio.savemat(date+'_'+str(size)+'_workers_'+str(n_epoch)+'_epochs_'+str(p)+'_class_range'+str(range_)+'_CENT.mat', mdic)
            elif W_type == 6:
                sio.savemat(date+'_'+str(size)+'_workers_'+str(n_epoch)+'_epochs_'+str(p)+'_class_range'+str(range_)+'_Fully-connected.mat', mdic)
            else:
                sio.savemat(date+'_'+str(size)+'_workers_'+str(n_epoch)+'_epochs_'+str(p)+'_class_range'+str(range_)+'_No-consensus.mat', mdic)

print('size =',size)
print('p =',p)
print('range_ =',range_)
print('W_type =',W_type)
print('bsz =',bsz)
print('n_epoch =',n_epoch)
print('batch_test_accuracy',batch_test_accuracy)
    
   








