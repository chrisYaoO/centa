import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms

from random import Random
import scipy.io as sio
import numpy as np
import argparse
import time
import os, pathlib


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


def partition_dataset(file_path):
    """ Partitioning MNIST """
    """ Assuming we have 2 replicas, then each process will have a train_set of 60000 / 2 = 30000 samples. We also divide the batch size by the number of replicas in order to maintain the overall batch size of 128."""
    """CIFAR10, EMNIST,Fashion-MNIST"""
    # print('  start loading dataset')
    # start_time = time.time()
    dataset = datasets.MNIST(
        root=file_path,
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    # print('  dataset',dataset)
    size = dist.get_world_size()
    # print('  size',size)
    bsz = 64  # int(256*50/ float(size))#int(256)#int(128 / float(size))
    # print('  bsz',bsz)
    partition_sizes = [1.0 / size for _ in range(size)]
    # print('  partition_sizes',partition_sizes)
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    # print('  train_set',len(train_set))
    return train_set, bsz


def partition_dataset_test(filepath):
    """ Partitioning MNIST """
    """ Assuming we have 2 replicas, then each process will have a train_set of 60000 / 2 = 30000 samples. We also divide the batch size by the number of replicas in order to maintain the overall batch size of 128."""
    """CIFAR10, EMNIST,Fashion-MNIST"""
    # print('  start loading dataset')
    # start_time = time.time()
    dataset = datasets.MNIST(
        root=filepath,
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    # print('  dataset',dataset)
    # size = dist.get_world_size()
    # #print('  size',size)

    # partition_sizes = [1.0 / size for _ in range(size)]
    # #print('  partition_sizes',partition_sizes)

    bsz = 10000  # int(128 / float(size))
    # #print('  bsz',bsz)

    # partition = DataPartitioner(dataset, partition_sizes)
    # partition = partition.use(dist.get_rank())
    test_set = torch.utils.data.DataLoader(
        dataset, batch_size=bsz, shuffle=True)
    # print('  test_set',len(test_set))
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


def average_gradients(model, W):
    """ Gradient averaging. """

    ID = dist.get_rank()
    size = int(dist.get_world_size())

    for param in model.parameters():
        # 1. 在「参数所在的 GPU」上建临时缓冲
        tensor_list = [torch.empty_like(param.data)
                       for _ in range(size)]
        dist.all_gather(tensor_list, param.data)

        param.data = sum(W[ID][i] * tensor_list[i] for i in range(size))


def output_all(flag):
    if flag:
        return range(dist.get_world_size())
    else:
        return [0]


if __name__ == '__main__':
    print("cwd =", pathlib.Path.cwd())

    print("torch.version.cuda  :", torch.version.cuda)  # 编译期用的 CUDA 主版本号
    print("torch.version.git   :", torch.version.git_version)  # 方便确认 wheel 来源
    print("cuda.is_available() :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Driver capability :", torch.cuda.get_device_capability(0))
        print("Device name       :", torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--backend", type=str, default="nccl")  # GPU: nccl, CPU: gloo
    parser.add_argument("--w_type", type=int, default=5)  # 1~7
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f'rank={rank}, world_size={world_size}', flush=True)
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    dist.init_process_group(backend=args.backend, init_method="env://")

    print('init success')
    # print(f'second: rank={dist.get_rank()}, worldsize={dist.get_world_size()}', flush=True)

    W_type = args.w_type
    n_epoch = args.epochs
    range_ = 60
    case = 1
    size = 4

    filename = 'fed/' + 'CENT_solutions_iter1_' + str(size) + 'workers_range' + str(range_) + '.mat'
    print(filename)

    if W_type == 1:  # md
        W = sio.loadmat(filename)['W_md']
    elif W_type == 2:  # mh
        W = sio.loadmat(filename)['W_mh']
    elif W_type == 3:  # bc
        W = sio.loadmat(filename)['W_bc']
    elif W_type == 4:  # fdla
        W = sio.loadmat(filename)['W_fdla']
    elif W_type == 5:  # cent
        W = sio.loadmat(filename)['W_cent']
    elif W_type == 6:  # fully connected
        W = np.ones((int(size), int(size))) * (1 / size)
    else:  # no communication
        W = np.identity(size)

    print('w read success')
    print(W)

    W = W.to(device)

    torch.manual_seed(10 * dist.get_rank())  # 4321)#1234)

    # here i iid data, can also use the non-iid case as in the simulation code
    file_path = 'fed/datasets'
    train_set, bsz = partition_dataset(file_path)
    test_set, test_bsz = partition_dataset_test(file_path)
    # print(bsz,test_bsz)
    print('dataset loaded')

    model = create_lenet().to(device)  # Net() #LeNet()#MLP()#AlexNet()#
    # model = model
    # model = model.cuda(rank)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    cec = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    # num_batches = ceil(len(train_set.dataset) / float(bsz))
    batch_loss = []
    batch_accuracy = []
    run_time = []
    run_comp_time = []

    batch_test_loss = []
    batch_test_accuracy = []
    run_test_time = []

    iter_count = 0
    flag = 1
    print('start training:', rank)
    for epoch in range(n_epoch):

        if dist.get_rank() == output_all(flag):
            print('epoch =', epoch)

        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_idx = 0

        for data, target in train_set:
            batch_idx += 1

            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = cec(output, target)

            epoch_loss += loss.item()  # data[0]
            loss.backward()

            average_gradients(model, W)

            optimizer.step()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            """append data"""
            batch_loss.append(epoch_loss / (batch_idx + 1))
            batch_accuracy.append(100. * correct / total)

            if dist.get_rank() == output_all(flag):
                if batch_idx % int(len(train_set) / 5) == 0:
                    print(batch_idx, len(train_set), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                          % (epoch_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            iter_count += 1

        if dist.get_rank() == output_all(flag):
            """ Test """
            test_epoch_loss = 0.0
            test_correct = 0
            test_total = 0
            test_batch_idx = 0

            # test_start_time = time.time()

            for data, target in test_set:
                test_batch_idx += 1

                data, target = Variable(data), Variable(target)

                output = model(data)
                loss = F.nll_loss(output, target)
                test_epoch_loss += loss.item()  # data[0]
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
                print(test_batch_idx, len(test_set), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_epoch_loss / (test_batch_idx + 1), 100. * test_correct / test_total, test_correct,
                         test_total))

            """append data"""
            batch_test_loss.append(test_epoch_loss / (test_batch_idx + 1))
            batch_test_accuracy.append(100. * test_correct / test_total)

    if dist.get_rank() == output_all(flag):
        mdic = {'batch_training_loss': batch_loss,
                'batch_training_accuracy': batch_accuracy,
                'run_training_time': run_time,
                'run_comp_time': run_comp_time,
                'batch_test_loss': batch_test_loss,
                'batch_test_accuracy': batch_test_accuracy,
                'run_test_time': run_test_time,
                'n_epoch': n_epoch,
                'worker_ID': dist.get_rank()}
        for k, v in mdic.items():
            print(k, v)
        # if W_type == 1:
        #     sio.savemat("worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(
        #         n_epoch) + "_epochs_Max-degree_range" + str(range_) + str(case) + ".mat", mdic)
        # elif W_type == 2:
        #     sio.savemat("worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(
        #         n_epoch) + "_epochs_Metropolis_range" + str(range_) + str(case) + ".mat", mdic)
        # elif W_type == 3:
        #     sio.savemat("worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(
        #         n_epoch) + "_epochs_Best-constant_range" + str(range_) + str(case) + ".mat", mdic)
        # elif W_type == 4:
        #     sio.savemat(
        #         "worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(n_epoch) + "_epochs_FDLA_range" + str(
        #             range_) + str(case) + ".mat", mdic)
        # elif W_type == 5:
        #     sio.savemat(
        #         "worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(n_epoch) + "_epochs_CENT_range" + str(
        #             range_) + str(case) + "_static.mat", mdic)
        # elif W_type == 6:
        #     sio.savemat("worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(
        #         n_epoch) + "_epochs_Fully-connected_range" + str(range_) + str(case) + ".mat", mdic)
        # else:
        #     sio.savemat(
        #         "worker_" + str(dist.get_rank()) + "_of_" + str(size) + '_' + str(n_epoch) + "_epochs_No-consensus.mat",
        #         mdic)
