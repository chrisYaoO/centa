from train_dist import *
import numpy as np
import matplotlib.pyplot as plt


# plt.plot(timeline, train_acc_list, label='Train Accuracy')
# plt.plot(timeline, test_acc_list, label='Test Accuracy')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy vs. Time')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

p=6
bsz=4096

file_path='data/'
dataset = datasets.MNIST(
    root=file_path,
    train=True,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

N_class = 10
world_size = 4
rank = 2

""" create 10*6000 index list"""
labels = dataset.targets
index_pos_list = [(labels == i).nonzero(as_tuple=True)[0].tolist() for i in range(N_class)]

index_list = []
chunk_len = len(dataset) // world_size // p

for _ in range(world_size):
    y, z = 0, 0
    worker_idx = []
    for _ in range(p):
        start = z * chunk_len
        end = (z + 1) * chunk_len
        worker_idx.extend(index_pos_list[y][start:end])
        y = (y + 1) % 10
        if y == 0:
            z += 1
    index_list.append(worker_idx)

partition = DataPartitioner_niid(dataset, index_list)
subset = partition.use(rank)
loader = torch.utils.data.DataLoader(
    subset, batch_size=bsz, shuffle=True, drop_last=False
)
print(len(subset))
print(len(loader))
