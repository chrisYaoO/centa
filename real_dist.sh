#!/bin/bash
#SBATCH --job-name=real_dist
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=3G
#SBATCH --output=%x_%j_%t.out

set -euo pipefail                # print and exit immediately after error
set -x


source venv/bin/activate
#module load cuda/12.2


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500               # keep port available
export WORLD_SIZE=$SLURM_NTASKS

export TORCH_DIST_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1     # C++
export CUDA_LAUNCH_BLOCKING=1          # GPU error

export NCCL_DEBUG=INFO                  # NCCL
export NCCL_ASYNC_ERROR_HANDLING=1     # sync error
export NCCL_SOCKET_IFNAME=ib0      # ← 换成集群实际接口
export NCCL_P2P_LEVEL=SYS          # 允许跨节点 P2P
export NCCL_IB_TIMEOUT=22          # 避免大 RTT 时超时 (可选)

export PYTHONFAULTHANDLER=1            # python error
export PYTHONUNBUFFERED=1





srun --label python -m torch.distributed.run \
     --nnodes $SLURM_JOB_NUM_NODES \
     --nproc_per_node 1 \
     --node_rank $SLURM_NODEID \
     --rdzv_backend=c10d \
     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     real_dist.py \
     --epochs 10 \
     --backend nccl \
     --w_type 5
