#!/bin/bash
#SBATCH --job-name=real_dist
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1            # 1 GPU per process
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=2G
#SBATCH --output=%x.out

module load cuda/12.2
source venv/bin/activate

# -------- 1. 生成 torchrun 必需的环境变量 --------
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500               # keep port available
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONUNBUFFERED=1


# -------- 2. 启动训练 --------
srun --label python -m torch.distributed.run \
     --nnodes $SLURM_JOB_NUM_NODES \
     --nproc_per_node 1 \
     --node_rank $SLURM_NODEID \
     --master_addr $MASTER_ADDR \
     --master_port $MASTER_PORT \
     real_dist.py \
     --epochs 60 \
     --backend nccl \
     --w_type 5
