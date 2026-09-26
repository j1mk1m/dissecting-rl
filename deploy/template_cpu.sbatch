#!/bin/bash
#SBATCH --partition cpu
#SBATCH --time 0-04:00 # set time
#SBATCH --job-name=#NAME

#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log

source ~/.bashrc
conda activate osft

# Run bash script
bash #SCRIPT
