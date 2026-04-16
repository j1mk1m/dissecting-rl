#!/bin/bash
#SBATCH --partition general
#SBATCH --time 1-23:00
#SBATCH --job-name=grpo_mask
#SBATCH --chdir=/home/gyeongwk/compositional-generality

#SBATCH --gres=gpu:8
#SBATCH --mem=512GB
#SBATCH --cpus-per-task=8

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

export RAY_TMPDIR=/data/user_data/gyeongwk/ray_tmp

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate osft

bash experiments/osft/grpo_mask.sh
