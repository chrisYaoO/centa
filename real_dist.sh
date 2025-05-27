#!/bin/bash
#SBATCH --job-name=real_dist
#SBATCH --partition=gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1            # 每块 GPU 起 1 个进程
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.out

module load cuda/12.2
source activate fed

# -------- 1. 生成 torchrun 必需的环境变量 --------
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500               # 确保端口空闲
export WORLD_SIZE=$SLURM_NTASKS

# -------- 2. 启动训练 --------
srun torchrun \
     --nnodes $SLURM_JOB_NUM_NODES \   # =8
     --nproc_per_node 1 \              # 每台 1 进程
     --node_rank $SLURM_NODEID \
     --master_addr $MASTER_ADDR \
     --master_port $MASTER_PORT \
     /home/$USER/projects/fed/train_dist_multi.py \
     --epochs 60 \
     --backend nccl \
     --w_type 5         # 5 = CENT；按需修改
