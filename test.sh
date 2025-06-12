#!/bin/bash
#SBATCH --job-name=ucx_test
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=1G
#SBATCH --output=test.out

module load StdEnv/2023
module load cuda/12.2
module load openmpi/4.1.5
#module load ucc-cuda/1.2.0
#module load ucx-cuda/1.14.1
#module load gdrcopy/2.3.1
#module load nccl/2.18.3
source venv/bin/activate

srun -N2 -n2 --mpi=pmi2 python test.py





