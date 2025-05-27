#!/bin/bash
#SBATCH --account=def-benliang
#SBATCH --job-name=cifar-1gpu
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --output=test.out

source venv/bin/activate

python -u train_test.py
