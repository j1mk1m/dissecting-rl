#!/bin/bash
#SBATCH --partition cpu
#SBATCH --time 1-23:59 # set time
#SBATCH --job-name=#NAME

#SBATCH --mem=100GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=/home/${HF_USER:-anonymous}/logs/%x-%j.out # output log
#SBATCH --error=/home/${HF_USER:-anonymous}/logs/%x-%j.err # error log

source ~/.bashrc

# Run bash script
modal run experiments/string/run_modal.py --script-path #SCRIPT
