#!/bin/bash
#SBATCH --account=def-benliang
#SBATCH --job-name=comm-test
#SBATCH --nodes=2                 # 两台机器
#SBATCH --gres=gpu:v100:1         # 每台 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out

source venv/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_BLOCKING_WAIT=1       # 出错时容易定位
export NCCL_IB_DISABLE=0          # 确保走 InfiniBand

# 节点 × GPU
srun --label python -m torch.distributed.run \
     --nnodes=$SLURM_JOB_NUM_NODES \
     --nproc_per_node=1 \
     --rdzv_backend=c10d \
     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     comm_test.py --size_mb 100 --iters 200
