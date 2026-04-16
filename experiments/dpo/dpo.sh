#!/usr/bin/env bash
set -euo pipefail
set -x

# Example usage:
#   bash experiments/dpo/dpo.sh
#   NGPUS=4 VISIBLE_DEVICES="0,1,2,3" bash experiments/dpo/dpo.sh

export HYDRA_FULL_ERROR=1

EXP_NAME="dpo-example"

# -------- Editable paths --------
TRAIN_FILE="data/string_task/dpo/train.parquet"
VAL_FILE="data/string_task/dpo/test.parquet"
MODEL_PATH=${MODEL_PATH:-"gyeongwk/stage1-rft"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/${EXP_NAME}"}

# -------- Hardware --------
NGPUS=${NGPUS:-4}                          # Set 4 or 8 for your node.
VISIBLE_DEVICES=${VISIBLE_DEVICES:-"0,1,2,3"}

# -------- Training --------
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
MICRO_BATCH_PER_GPU=${MICRO_BATCH_PER_GPU:-1}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-4096}
LR=${LR:-5e-7}
BETA=${BETA:-0.1}
EPOCHS=${EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-25}

CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
torchrun \
  --standalone \
  --nproc_per_node="${NGPUS}" \
  src/dpo/dpo_trainer.py \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${GLOBAL_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_PER_GPU}" \
  data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  data.max_length="${MAX_TOKEN_LEN_PER_GPU}" \
  data.pad_mode=no_padding \
  data.prompt_key=prompt \
  data.chosen_key=chosen \
  data.rejected_key=rejected \
  model.path="${MODEL_PATH}" \
  optim.lr="${LR}" \
  dpo.beta="${BETA}" \
  trainer.logger="['console','wandb']" \
  trainer.project_name=dpo \
  trainer.experiment_name=${EXP_NAME} \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.total_epochs="${EPOCHS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.reference_mode=auto

