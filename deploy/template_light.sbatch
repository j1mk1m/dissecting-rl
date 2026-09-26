#!/bin/bash
#SBATCH --partition general
#SBATCH --time 0-01:00 # set time
#SBATCH --job-name=#NAME

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log

source ~/.bashrc
conda activate osft

# Run bash script
bash #SCRIPT
