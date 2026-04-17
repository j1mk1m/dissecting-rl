#!/bin/bash
#SBATCH --partition flame-earlybirds
#SBATCH --qos earlybird_qos
#SBATCH --time 1-23:00 # set time
#SBATCH --job-name=stage1_rft
#SBATCH --chdir=/home/gyeongwk/RL-Compositionality # working directory

#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rl_comp

export PROJECT_NAME=string-task
export EXPERIMENT_NAME=stage1-rft
export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export NNODES=1
export SP_SIZE=1
export MAX_LENGTH=3072
export TRAIN_FILES=data/string_task/stage1_level1/rft_data/train.parquet 
export VAL_FILES=data/string_task/stage1_level1/rft_data/test.parquet 
export BATCH_SIZE=128
export EPOCHS=2
export SAVE_DIR=/data/user_data/gyeongwk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash experiments/sft/template.sh
