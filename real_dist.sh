#!/bin/bash
#SBATCH --job-name=real_dist
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=%x_%j_%N_%t.out

set -euo pipefail                # 出错立即退出并打印
set -x


source venv/bin/activate
#module load cuda/12.2


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500               # keep port available
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONUNBUFFERED=1
export TORCH_DIST_DEBUG=DETAIL          # 打开 elastic 异常追踪
export TORCH_SHOW_CPP_STACKTRACES=1     # C++ 栈
export NCCL_DEBUG=INFO                  # NCCL 握手细节
export PYTHONFAULTHANDLER=1


srun --label python -m torch.distributed.run \
     --nnodes $SLURM_JOB_NUM_NODES \
     --nproc_per_node 1 \
     --node_rank $SLURM_NODEID \
     --rdzv_backend=c10d \
     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     real_dist.py \
     --epochs 60 \
     --backend nccl \
     --w_type 5
