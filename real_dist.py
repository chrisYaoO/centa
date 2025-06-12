import os

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


def consensus_average(model, W: torch.Tensor, link_rate_bps, option='mpi'):
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
        if link_rate_bps is not None:
            bytes_total = flat.numel() * flat.element_size() * len(nbr_ids)
            time.sleep(bytes_total * 8 / link_rate_bps)
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
        dtype = MPI.FLOAT if flat.dtype == torch.float32 else MPI.DOUBLE

        # find out connected neighbors(non-zero in W) and create buffer for each
        nbr_ids = [j for j, w in enumerate(w_row) if j != rank and w != 0]
        recv_bufs = {j: torch.empty_like(flat) for j in nbr_ids}

        # CUDA-aware P2P
        count = flat.numel()
        reqs = []
        for j in nbr_ids:
            reqs.append(comm.Isend([flat.data, count, dtype], dest=j, tag=_global_tag))
            logger.debug(f'sending from {rank} to {j}, tag: {_global_tag},count: {count}')
            reqs.append(comm.Irecv([recv_bufs[j], count, dtype], source=j, tag=_global_tag))
            logger.debug(f'{rank} recv from {j}, tag: {_global_tag},count: {count}')

        # logger.debug('waiting...')
        MPI.Request.Waitall(reqs)
        # logger.debug('synchronizing...')
        torch.cuda.synchronize(device)
        # logger.debug("yes!")

        # # bandwidth limit(in time)
        if link_rate_bps is not None:
            bytes_total = flat.numel() * flat.element_size() * len(nbr_ids)
            time.sleep(bytes_total * 8 / link_rate_bps)

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


def train_epoch(loader, W_gpu, link_rate_bps):
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0

    for data, target in loader:  # mini-batch loop
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        output = model(data)
        loss = cec(output, target)

        loss.backward()

        consensus_average(model, W_gpu, link_rate_bps=None)
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


def load_weight_matrix(filename, W_type=5):
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

    return W


if __name__ == '__main__':
    # parse param
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--backend", type=str, default="nccl")  # GPU: nccl, CPU: gloo
    parser.add_argument("--w_type", type=int, default=5)  # 1~7
    parser.add_argument("--output", type=str, default='info')
    args = parser.parse_args()
    # set logger
    logger = setup_logger(args.output)
    # set cuda
    logger.debug("cwd =", pathlib.Path.cwd())
    logger.debug("torch.version.cuda  :", torch.version.cuda)
    logger.debug("torch.version.git   :", torch.version.git_version)
    logger.debug("cuda.is_available() :", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.debug("Driver capability :", torch.cuda.get_device_capability(0))
        logger.debug("Device name       :", torch.cuda.get_device_name(0))
    else:
        logger.warning('cuda not available!!!!')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('device: ', device)

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
    case = 1
    size = 4
    flag = 0

    filename = 'fed/' + 'CENT_solutions_iter1_' + str(size) + 'workers_range' + str(range_) + '.mat'
    W = load_weight_matrix(filename, W_type=W_type)

    # W = [[0.5, 0.5], [0.5, 0.5]]
    # W = [[1 / 3, 1 / 3, 0, 1 / 3],
    #      [1 / 3, 1 / 3, 1 / 3, 0],
    #      [0, 1 / 3, 1 / 3, 1 / 3],
    #      [1 / 3, 0, 1 / 3, 1 / 3]]

    logger.debug('w read success')
    logger.debug(W)

    # BW = 20 * 1e6  # 20 Mb/s  -->  20 MHz(≈1 bps/Hz)
    BW = None

    W_gpu = torch.tensor(W, dtype=torch.float32,
                         device=torch.cuda.current_device(),
                         requires_grad=False)

    # training prepration
    torch.manual_seed(10 * dist.get_rank())

    # load MNIST dataset
    # here i iid data, can also use the non-iid case as in the simulation code
    file_path = 'fed/datasets'
    train_set, bsz = partition_dataset(file_path, bsz=4096)
    test_set, test_bsz = partition_dataset_test(file_path, bsz=10000)
    logger.debug('dataset loaded')

    # model settings
    model = create_lenet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    cec = nn.CrossEntropyLoss().to(device)

    timeline = []
    train_acc_list = []
    test_acc_list = []

    start_time = time.time()

    logger.info('start training:', rank)
    for epoch in range(n_epoch):
        train_loss, train_acc = train_epoch(train_set, W_gpu, BW)
        val_loss, val_acc = evaluate(test_set)

        if rank == 0:
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
        df.to_csv('data/accuracy_vs_time.csv', index=False)
        logger.info('data saved')

    # end process in case memory leak
    dist.destroy_process_group()
    logger.info('completed')
