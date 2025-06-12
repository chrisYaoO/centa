import os
os.environ.setdefault("MPI4PY_RC_CUDA", "yes")   # 环境兜底

# ---------- ① 先完整导入 mpi4py ----------
import mpi4py
from mpi4py import rc

print("mpi4py version :", mpi4py.__version__)
print("has rc.cuda    :", hasattr(rc, "cuda"))

# ---------- ② 关闭 auto-init，打开 CUDA-aware ----------
rc.initialize = False          # 禁止 mpi4py 自动调用 MPI.Init
if hasattr(rc, "cuda"):        # 只有 3.1+ 版本才有 cuda 属性
    rc.cuda = True

# ---------- ③ 现在再导入 MPI ----------
from mpi4py import MPI
MPI.Init()                     # 手动初始化（必须！）
comm = MPI.COMM_WORLD

import torch, sys

rank, size = comm.Get_rank(), comm.Get_size()
if size < 2:
    if rank == 0:
        print("Need at least 2 ranks for the smoke test.")
    sys.exit()

torch.cuda.set_device(rank % torch.cuda.device_count())
N = 256_000
send = torch.full((N,), float(rank + 1), device="cuda")
recv = torch.empty_like(send)
dtype = MPI.FLOAT

peer = (rank + 1) % size
tag = 777

req_s = comm.Isend([send, dtype], dest=peer, tag=tag)
req_r = comm.Irecv([recv, dtype], source=peer, tag=tag)
print('waiting')
MPI.Request.Waitall([req_s, req_r])

err = (send.mean() + recv.mean()).item()
print(f"Rank {rank}: send mean={send.mean().item():.1f}, "
      f"recv mean={recv.mean().item():.1f}, peer={peer}")

MPI.Finalize()
