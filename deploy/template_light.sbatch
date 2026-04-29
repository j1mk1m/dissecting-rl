#!/bin/bash
#SBATCH --partition general
#SBATCH --time 0-01:00 # set time
#SBATCH --job-name=#NAME
#SBATCH --chdir=/home/gyeongwk/compositional-generality # working directory

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

source ~/.bashrc
conda activate compositional

# Run bash script
bash #SCRIPT
