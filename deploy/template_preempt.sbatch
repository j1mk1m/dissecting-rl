#!/bin/bash
#SBATCH --partition preempt
#SBATCH --time 1-23:00 # set time
#SBATCH --job-name=#NAME
#SBATCH --chdir=/home/gyeongwk/compositional-generality # working directory

#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

source ~/.bashrc
conda activate osft

# Run bash script
bash #SCRIPT
