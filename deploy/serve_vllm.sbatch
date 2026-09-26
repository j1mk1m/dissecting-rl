#!/bin/bash
#SBATCH --partition general
#SBATCH --time 1-23:59 # set time
#SBATCH --job-name=vllm

#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log

source ~/.bashrc
conda activate osft

# Run bash script
bash experiments/util/serve_vllm.sh


