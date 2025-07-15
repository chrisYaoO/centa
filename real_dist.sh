#!/bin/bash
#SBATCH --job-name=real_dist
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=2G
#SBATCH --output=%x_%j_%t.out

set -euo pipefail                # print and exit immediately after error
set -x

module load cuda/12.2
module load openmpi/4.1.5
module load mpi4py/4.0.3
module load python/3.11.5
module load ucx-cuda/1.14.1
module load ucc-cuda/1.2.0
source venv/bin/activate



export UCX_TLS=rc,cuda_copy,cuda_ipc,tcp
export OMPI_MCA_pml=ucx  # force mpi to use ucx
export OMPI_MCA_osc=ucx
export UCX_NET_DEVICES=all
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500               # keep port available


export TORCH_DIST_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1     # C++
export CUDA_LAUNCH_BLOCKING=1          # GPU error

#export NCCL_DEBUG=INFO                  # NCCL
#export NCCL_ASYNC_ERROR_HANDLING=1     # sync error
#export NCCL_SOCKET_IFNAME=ib0      # port
#export NCCL_P2P_LEVEL=SYS          # allow p2p
#export NCCL_IB_TIMEOUT=22          # avoid rtt

export PYTHONFAULTHANDLER=1            # python error
export PYTHONUNBUFFERED=1

#srun --mpi=pmi2 python real_dist.py \
#     --batch_size 4096 \
#     --epochs 20 \
#     --backend nccl \
#     --w_type 5 \
#     --output debug \
#     --model lenet

PARAMS_LIST=(
#    "--batch_size 256 --epochs 100 --backend nccl --w_type 8 --output info --model lenet --p 10"
#    "--batch_size 256 --epochs 100 --backend nccl --w_type 5 --output info --model lenet --p 10"
#    "--batch_size 256 --epochs 100 --backend nccl --w_type 4 --output info --model lenet --p 10"
#    "--batch_size 256 --epochs 100 --backend nccl --w_type 3 --output info --model lenet --p 10"
#    "--batch_size 256 --epochs 100 --backend nccl --w_type 2 --output info --model lenet --p 10"
#    "--batch_size 256 --epochs 100 --backend nccl --w_type 1 --output info --model lenet --p 10"
#    "--batch_size 256 --epochs 100 --backend nccl --w_type 8 --output info --model lenet --p 6"
    "--batch_size 256 --epochs 100 --backend nccl --w_type 5 --output info --model lenet --p 6"
    "--batch_size 256 --epochs 100 --backend nccl --w_type 4 --output info --model lenet --p 6"
    "--batch_size 256 --epochs 100 --backend nccl --w_type 3 --output info --model lenet --p 6"
    "--batch_size 256 --epochs 100 --backend nccl --w_type 2 --output info --model lenet --p 6"
    "--batch_size 256 --epochs 100 --backend nccl --w_type 1 --output info --model lenet --p 6"
)

for param in "${PARAMS_LIST[@]}"; do
    echo "======= $(date): running $param ======="
    srun --mpi=pmi2 python real_dist.py $param
done

echo "All runs finished!"