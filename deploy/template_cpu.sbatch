#!/bin/bash
#SBATCH --partition cpu
#SBATCH --time 0-04:00 # set time
#SBATCH --job-name=#NAME
#SBATCH --chdir=/home/gyeongwk/compositional-generality # working directory

#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

source ~/.bashrc
conda activate osft

# Run bash script
bash #SCRIPT
