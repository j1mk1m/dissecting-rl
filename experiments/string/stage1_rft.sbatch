#!/bin/bash
#SBATCH --partition general
#SBATCH --time 1-23:00 # set time
#SBATCH --job-name=stage1_rft

#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log

source ~/.bashrc
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
export SAVE_DIR=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash experiments/_archive/sft/template.sh
