#!/bin/bash
#SBATCH --partition cpu
#SBATCH --time 1-23:59 # set time
#SBATCH --job-name=#NAME
#SBATCH --chdir=/home/gyeongwk/compositional-generality # working directory

#SBATCH --mem=100GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=/home/gyeongwk/logs/%x-%j.out # output log
#SBATCH --error=/home/gyeongwk/logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

source ~/.bashrc

# Run bash script
modal run experiments/run_modal.py --script-path #SCRIPT
