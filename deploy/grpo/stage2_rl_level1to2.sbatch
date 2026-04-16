#!/bin/bash
#SBATCH --partition flame-earlybirds
#SBATCH --qos earlybird_qos
#SBATCH --time 1-23:00 # set time
#SBATCH --job-name=stage2_rl_level1to2
#SBATCH --chdir=/home/gyeongwk/RL-Compositionality # working directory

#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

export RAY_TMPDIR=/data/user_data/gyeongwk/ray_tmp

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rl_comp

bash bash/section41_42/stage2_rl_level1to2.sh
