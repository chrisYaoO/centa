from train_dist import *
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

csv_paths = glob.glob("data/*.csv")

W_type = ['MD', 'MH', 'BC', 'FDLA', 'CENT', 'FC', 'NC', 'CENT-A']
pattern = re.compile(
    r"""
    ^data
    (?P<model>.+?)_                       # model_name  (贪婪最小匹配到下一个 "_")
    (?P<dataset>.+?)_
    N_(?P<N>\d+)_                         # world_size
    p_(?P<p>\d+)_                         # p
    w_(?P<w>\d+)_                         # W_type
    e_(?P<e>\d+)_                         # n_epoch
    b_(?P<b>\d+)                          # batch_size
    \.csv$
    """,
    re.VERBOSE,
)

p_list = [6, 10]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for path in csv_paths:
    m = pattern.match(path)
    data = pd.read_csv(path)
    if not m:
        print(f"[warn] 跳过非标准文件名: {path}")
        continue

    # 把命名参数抓出来
    meta = m.groupdict()
    # 数字都转 int
    for key in ("N", "p", "e", "b", 'w'):
        meta[key] = int(meta[key])
    print(meta)
    if meta['p'] == 6:
        axes[0].plot(data['time'], data['test_acc'], label=W_type[meta['w'] - 1])
    else:
        axes[1].plot(data['time'], data['test_acc'], label=W_type[meta['w'] - 1])
for ax in axes:
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Test Accuracy')
    ax.grid(True)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 3500)
axes[0].set_title('p=6')
axes[1].set_title('p=10')
# plt.xlabel('time')
# plt.ylabel('global_test_acc')
fig.tight_layout()
title = 'MNIST_Lenet_100_epochs_1024_bsz_20_workers'
fig.suptitle(title)
plt.show()
fig.savefig('results/'+title+'.png')
