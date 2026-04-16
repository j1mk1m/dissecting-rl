#!/bin/bash
#SBATCH --partition general
#SBATCH --time 1-23:00 # set time
#SBATCH --job-name=grpo
#SBATCH --chdir=/home/gyeongwk/compositional-generality # working directory

#SBATCH --gres=gpu:4
#SBATCH --mem=512GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

export RAY_TMPDIR=/data/user_data/gyeongwk/ray_tmp

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rl_comp

export MODEL_PATH=gyeongwk/stage1-rft
export PROJECT_NAME=string-task
export EXPERIMENT_NAME=grpo-test
export TRAIN_FILES="['data/string_task/stage2_level2/train.parquet']"
export VAL_FILES="['data/string_task/stage2_level1to8/test.parquet']"
export NNODES=1
export SAVE_DIR=/data/user_data/gyeongwk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

bash experiments/grpo/grpo_template.sh
