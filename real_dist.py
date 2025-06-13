import os

from objax.zoo.convnet import ConvNet

os.environ.setdefault("MPI4PY_RC_CUDA", "yes")

import mpi4py
from mpi4py import rc

print("mpi4py version :", mpi4py.__version__)
# logger.debug("has rc.cuda    :", hasattr(rc, "cuda"))  # false

rc.initialize = False
# if hasattr(rc, "cuda"):
#     rc.cuda = True

from mpi4py import MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
print(f"MPI rank={rank}, world={world_size}")

_global_tag = 0

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
import pandas as pd
import logging


def setup_logger(log_level="info"):
    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()

    if log_level.lower() == "debug":
        handler.setLevel(logging.DEBUG)
    elif log_level.lower() == "info":
        handler.setLevel(logging.INFO)
    else:
        raise ValueError("Unsupported log level. Use 'info' or 'debug'.")

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger


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
        self.partitions = indexes

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset_niid(file_path: str, p: int, bsz: int):
    """ Partitioning MNIST """

    dataset = datasets.MNIST(
        root=file_path,
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    N_class = 10
    size = dist.get_world_size()

    """ create 10*6000 index list"""
    labels = dataset.targets.tolist()
    index_pos_list = [(labels == i).nonzero(as_tuple=True)[0].tolist() for i in range(N_class)]

    x = 0
    y = 0
    z = 0
    index_list = []
    N_ = int((len(dataset) / size) / p)
    for i in range(size):
        index_list_worker = []
        for j in range(p):
            index_list_worker = index_list_worker + index_pos_list[y][z * N_:(z + 1) * N_]
            if y == N_class - 1:  # move to the next class
                y = 0
                z += 1
            else:
                y += 1
        x += 1  # move to the next worker
        index_list.append(index_list_worker)

    partition = DataPartitioner_niid(dataset, index_list)
    train_set = []
    for i in range(size):
        partition_ = partition.use(i)
        train_set_ = torch.utils.data.DataLoader(
            partition_, batch_size=bsz, shuffle=True)
        # print('  train_set',len(train_set))
        train_set.append(train_set_)
    return train_set  # n by 60000/n matrixs


def partition_dataset(file_path, bsz):
    """ Partitioning MNIST """
    """ Assuming we have 2 replicas, then each process will have a train_set of 60000 / 2 = 30000 samples. We also divide the batch size by the number of replicas in order to maintain the overall batch size of 128."""
    """CIFAR10, EMNIST,Fashion-MNIST"""
    # logger.debug('  start loading dataset')
    # start_time = time.time()
    dataset = datasets.MNIST(
        root=file_path,
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    size = dist.get_world_size()
    # bsz = 64  # int(256*50/ float(size))#int(256)#int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    # logger.debug('  partition_sizes',partition_sizes)
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    # logger.debug('  train_set',len(train_set))
    return train_set, bsz


def partition_dataset_test(filepath, bsz):
    """ Partitioning MNIST """
    """ Assuming we have 2 replicas, then each process will have a train_set of 60000 / 2 = 30000 samples. We also divide the batch size by the number of replicas in order to maintain the overall batch size of 128."""
    """CIFAR10, EMNIST,Fashion-MNIST"""
    # logger.debug('  start loading dataset')
    # start_time = time.time()
    dataset = datasets.MNIST(
        root=filepath,
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    # logger.debug('  dataset',dataset)
    # size = dist.get_world_size()
    # #logger.debug('  size',size)

    # partition_sizes = [1.0 / size for _ in range(size)]
    # #logger.debug('  partition_sizes',partition_sizes)

    # bsz = 10000  # int(128 / float(size))
    # #logger.debug('  bsz',bsz)

    # partition = DataPartitioner(dataset, partition_sizes)
    # partition = partition.use(dist.get_rank())
    test_set = torch.utils.data.DataLoader(
        dataset, batch_size=bsz, shuffle=True)
    # logger.debug('  test_set',len(test_set))
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


class CNN_Net(nn.Module):
    """ Network architecture. """

    # summary(model,(3,28,28))
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
    # INPUT_DIM = 28 * 28
    # OUTPUT_DIM = 10
    def __init__(self):
        super().__init__()

        self.input_fc = nn.Linear(28 * 28, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, 10)

    def forward(self, x):
        # x = [batch size, height, width]

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        # x = [batch size, height * width]

        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))

        # h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return y_pred  # , h_2


# Define AlexNet network structure
class AlexNet(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(  # Input 1*28*28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*14*14
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*7*7
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128*7*7
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256*7*7
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*3*3
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


@torch.no_grad()
def average_gradients(model, W):
    """ Gradient averaging. """

    rank = dist.get_rank()
    world = dist.get_world_size()

    for p in model.parameters():
        if p.grad is None:  # 这层没有梯度就跳过
            continue

            # 收集所有节点的梯度到 buf（GPU 上）
        buf = [torch.empty_like(p.grad) for _ in range(world)]
        dist.all_gather(buf, p.grad)

        # 加权求和：gᵢ ← Σⱼ Wᵢⱼ · gⱼ
        p.grad[:] = sum(W_gpu[rank, j].item() * buf[j] for j in range(world))

    dist.barrier()


def consensus_average(model, W: torch.Tensor, B_ij: torch.Tensor, option='mpi'):
    if (option == 'nccl'):
        rank = dist.get_rank()
        world = dist.get_world_size()
        assert W.shape == (world, world)

        device = next(model.parameters()).device  # GPU 0, 1, ...
        w_row = W[rank].to(device)  # move current row to gpu

        # flatten grads
        grads = [p.grad.data.view(-1) for p in model.parameters()
                 if p.grad is not None]
        flat = torch.cat(grads)  # (d,)

        # find out connected neighbors(non-zero in W) and create buffer for each
        nbr_ids = [j for j, w in enumerate(w_row) if j != rank and w != 0]
        recv_bufs = {j: torch.empty_like(flat) for j in nbr_ids}

        # nccl p2p send/rec
        reqs = []
        for j in nbr_ids:
            reqs.append(dist.isend(flat, dst=j))
            reqs.append(dist.irecv(recv_bufs[j], src=j))
        for r in reqs:
            r.wait()

        torch.cuda.synchronize()

        # bandwidth limit(in time)
        if B_ij is not None:
            bytes_total = flat.numel() * flat.element_size() * len(nbr_ids)
            time.sleep(bytes_total * 8 / B_ij)
        # weighted aggreagation
        mixed = w_row[rank] * flat  # itself
        for j in nbr_ids:
            mixed += w_row[j] * recv_bufs[j]

        # normalize: mixed /= w_row.sum()

        # update the grad
        idx = 0
        for p in model.parameters():
            if p.grad is None: continue
            n = p.grad.data.numel()
            p.grad.data.copy_(mixed[idx:idx + n].view_as(p.grad))
            idx += n
    elif (option == 'mpi'):
        global _global_tag

        rank = comm.Get_rank()
        world = comm.Get_size()
        assert W.shape == (world, world)

        device = next(model.parameters()).device  # GPU 0, 1, ...
        w_row = W[rank].to(device)  # move current row to gpu

        # flatten grads
        grads = [p.grad.data.view(-1) for p in model.parameters()
                 if p.grad is not None]
        flat = torch.cat(grads)  # GPU Tensor
        count = flat.numel()
        bits_model = count * flat.element_size() * 8

        dtype = MPI.FLOAT if flat.dtype == torch.float32 else MPI.DOUBLE

        # find out connected neighbors(non-zero in W) and create buffer for each
        nbr_ids = [j for j, w in enumerate(w_row) if j != rank and w != 0]
        recv_bufs = {j: torch.empty_like(flat) for j in nbr_ids}

        # CUDA-aware P2P
        reqs = []
        for j in nbr_ids:
            reqs.append(comm.Isend([flat.data, count, dtype], dest=j, tag=_global_tag))
            # logger.debug(f'sending from {rank} to {j}, tag: {_global_tag},count: {count}')
            reqs.append(comm.Irecv([recv_bufs[j], count, dtype], source=j, tag=_global_tag))
            # logger.debug(f'{rank} recv from {j}, tag: {_global_tag},count: {count}')

        # simulate bandwidth delay
        if B_ij is not None:
            B_ij_row = B_ij[rank]
            tx_times = [bits_model / (B_ij_row[j] * 1e6) for j in nbr_ids]
            comm_latency = max(tx_times)
            time.sleep(comm_latency.item())
            logger.debug(f'delay: {comm_latency.item()}')

        # logger.debug('waiting...')
        MPI.Request.Waitall(reqs)
        # logger.debug('synchronizing...')
        torch.cuda.synchronize(device)
        # logger.debug("yes!")

        # # bandwidth limit(in time)

        # weighted aggreagation
        mixed = w_row[rank] * flat
        for j in nbr_ids:
            mixed += w_row[j] * recv_bufs[j]
        # normalize: mixed /= w_row.sum()

        # # update the grad and tag
        idx = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            n = p.grad.data.numel()
            p.grad.data.copy_(mixed[idx:idx + n].view_as(p.grad))
            idx += n

        _global_tag += 1
    else:
        raise ValueError('Unsupported option')


def train_epoch(loader, W_gpu, B_ij):
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0

    for data, target in loader:  # mini-batch loop
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        output = model(data)
        loss = cec(output, target)

        loss.backward()

        consensus_average(model, W_gpu, B_ij)
        logger.debug('consensus_average done')
        optimizer.step()

        # stats
        epoch_loss += loss.item() * target.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    avg_loss = epoch_loss / total
    acc = 100. * correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(data)
        loss = cec(output, target)

        loss_sum += loss.item() * target.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    avg_loss = loss_sum / total
    acc = 100. * correct / total
    return avg_loss, acc


def output_all(flag):
    if flag:
        return range(dist.get_world_size())
    else:
        return [0]


def load_matrix(filename, size: int, W_type=5):
    if W_type == 1:  # md
        W = sio.loadmat(filename)['W_md']
        B = sio.loadmat(filename)['B_md']
    elif W_type == 2:  # mh
        W = sio.loadmat(filename)['W_mh']
        B = sio.loadmat(filename)['B_mh']
    elif W_type == 3:  # bc
        W = sio.loadmat(filename)['W_bc']
        B = sio.loadmat(filename)['B_bc']
    elif W_type == 4:  # fdla
        W = sio.loadmat(filename)['W_fdla']
        B = sio.loadmat(filename)['B_fdla']
    elif W_type == 5:  # cent
        W = sio.loadmat(filename)['W_cent']
        B = sio.loadmat(filename)['B_cent']
    elif W_type == 6:  # fully connected
        W = np.ones((int(size), int(size))) * (1 / size)
        B = np.ones((int(size), int(size))) * 20 / (size * size)
    else:  # no communication
        W = np.identity(size)
        B = np.zeros(int(size) * int(size))

    B_ij = np.zeros((size, size))
    if np.count_nonzero(W) != len(B) + size:
        raise ValueError('size of W and B do not match')
    else:
        k = 0
        for i in range(size):
            for j in range(size):
                if W[i][j] != 0 and i != j:
                    B_ij[i][j] = B[k]
                    k += 1

    return W, B_ij


def create_model(model_name, rank):
    model_map = {
        'lenet': create_lenet,
        'alexnet': AlexNet,
        'cnn': CNN_Net,
        'net': Net,
        'MLP': MLP
    }

    if model_name not in model_map:
        raise ValueError(f'Model type "{model_name}" not supported.')

    model_fn = model_map[model_name]
    model = model_fn().to(device)
    global_model = model_fn().to(device) if rank == 0 else None

    return model, global_model


@torch.no_grad()
def update_global_model(local_model, global_model):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    for (name, param) in local_model.named_parameters():
        tmp = param.data.clone()

        dist.reduce(tmp, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            tmp /= world_size
            global_param = dict(global_model.named_parameters())[name]
            global_param.copy_(tmp)

    for (name, buf) in local_model.named_buffers():
        tmp = buf.data.clone()
        dist.reduce(tmp, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            tmp /= world_size
            global_buf = dict(global_model.named_buffers())[name]
            global_buf.copy_(tmp)


if __name__ == '__main__':
    # parse param
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--backend", type=str, default="nccl")  # GPU: nccl, CPU: gloo
    parser.add_argument("--w_type", type=int, default=5)  # 1~7
    parser.add_argument("--output", type=str, default='info')
    parser.add_argument("--model", type=str, default='lenet')
    parser.add_argument("--dataset", type=str, default='mnist')

    args = parser.parse_args()
    # set logger
    logger = setup_logger(args.output)
    # set cuda
    logger.debug("cwd =", pathlib.Path.cwd())
    logger.debug("torch.version.cuda: %s", torch.version.cuda)
    logger.debug("torch.version.git %s:", torch.version.git_version)
    logger.debug("cuda.is_available(): %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.debug("Driver capability: %s", torch.cuda.get_device_capability(0))
        logger.debug("Device name: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning('cuda not available!!!!')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('device: %s', device)

    buf = torch.zeros(10, device='cuda')

    # try:
    #     MPI.COMM_WORLD.Isend([buf, MPI.FLOAT], dest=(MPI.COMM_WORLD.Get_rank() + 1) % MPI.COMM_WORLD.Get_size(), tag=0)
    #     logger.debug("CUDA-aware send appears to work ")
    # except MPI.Exception as e:
    #     logger.debug("CUDA-aware send FAILED ✘ :", e)

    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    # logger.debug(f'rank={rank}, world_size={world_size}', flush=True)
    # master_addr = os.environ["MASTER_ADDR"]
    # master_port = os.environ["MASTER_PORT"]

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # init multi nodes
    dist.init_process_group(backend=args.backend, init_method="env://")

    logger.debug("DDP/NCCL init ok")

    # load weight matrix and put it on gpu
    W_type = args.w_type
    n_epoch = args.epochs
    range_ = 60
    size = 4
    p = 6 # num labels 1-10

    filename = 'fed/' + 'CENT_solutions_iter1_' + str(size) + 'workers_range' + str(range_) + '.mat'
    W, B_ij = load_matrix(filename, size, W_type)

    # W = [[0.5, 0.5], [0.5, 0.5]]
    # W = [[1 / 3, 1 / 3, 0, 1 / 3],
    #      [1 / 3, 1 / 3, 1 / 3, 0],
    #      [0, 1 / 3, 1 / 3, 1 / 3],
    #      [1 / 3, 0, 1 / 3, 1 / 3]]

    W_gpu = torch.tensor(W, dtype=torch.float32,
                         device=torch.cuda.current_device(),
                         requires_grad=False)

    eta = 4  # spectrum efficiency? bit/s/Hz（16-QAM≈4）
    B_ij = eta * B_ij
    # B_ij = None
    B_ij_gpu = torch.tensor(B_ij, dtype=torch.float32,
                            device=torch.cuda.current_device(),
                            requires_grad=False)

    logger.debug('mat read success')
    logger.debug([W, B_ij])

    # training prepration
    torch.manual_seed(10 * dist.get_rank())

    # load MNIST dataset
    # here i iid data, can also use the non-iid case as in the simulation code
    file_path = 'fed/datasets'
    train_set, bsz = partition_dataset_niid(file_path, p=6, bsz=4096)
    test_set, test_bsz = partition_dataset_test(file_path, bsz=10000)
    logger.debug('dataset loaded')

    # model settings
    model_name = args.model
    model, global_model = create_model(model_name, rank=rank)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    cec = nn.CrossEntropyLoss().to(device)

    timeline = []
    train_acc_list = []
    test_acc_list = []

    start_time = time.time()

    logger.info(f'start training:{rank}')
    for epoch in range(n_epoch):
        train_loss, train_acc = train_epoch(train_set, W_gpu, B_ij_gpu)

        update_global_model(model, global_model)

        if rank == 0:
            val_loss, val_acc = evaluate(test_set, model)
            elapsed_time = time.time() - start_time
            timeline.append(elapsed_time)
            train_acc_list.append(train_acc)
            test_acc_list.append(val_acc)

            logger.info(f"[Epoch {epoch:03d}] "
                        f"train loss {train_loss:.4f}  acc {train_acc:.2f}%  "
                        f"val loss {val_loss:.4f}  acc {val_acc:.2f}%")

    if rank == 0:
        df = pd.DataFrame({
            'time': timeline,
            'train_acc': train_acc_list,
            'test_acc': test_acc_list
        })
        df.to_csv(f'data/acc_vs_time_{model_name}_{args.dataset}_{world_size}_{p}_w_{W_type}_epoch_{n_epoch}.csv', index=False)
        logger.info('data saved')

    # end process in case memory leak
    dist.destroy_process_group()
    logger.info('completed')
