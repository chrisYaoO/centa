#!/bin/bash
#SBATCH --account=def-benliang
#SBATCH --job-name=cifar-1gpu
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --output=test.out
#SBATCH --mem=1G

source ../venv/bin/activate

export PYTHONUNBUFFERED=1

python -u train_test.py
