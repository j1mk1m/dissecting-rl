#!/bin/bash
#SBATCH --partition general
#SBATCH --time 0-23:00 # set time
#SBATCH --job-name=eval
#SBATCH --chdir=/home/gyeongwk/RL-Compositionality # working directory

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

# RFT setup
PROJECT_NAME=string-task
EXPERIMENT_NAME=eval

NNODES=1
NGPUS=4
N_SAMPLES=1
DATA_PATH=data/string_task/stage2_level1to8_ID/test.parquet
TEMPERATURE=0.0
PROMPT_LENGTH=1024
RESPONSE_LENGTH=8192
BATCH_SIZE=4

MODEL_NAME=stage2-rft
MODEL_PATH=gyeongwk/$MODEL_NAME

SAVE_PATH=eval/$MODEL_NAME-ID/rollout.parquet

python3 -m verl.trainer.main_generation \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=$NGPUS \
    data.path=$DATA_PATH \
    data.prompt_key=prompt \
    data.n_samples=$N_SAMPLES \
    data.output_path=$SAVE_PATH \
    data.batch_size=$BATCH_SIZE \
    model.path=$MODEL_PATH \
    +model.trust_remote_code=True \
    rollout.name=vllm \
    rollout.temperature=$TEMPERATURE \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=$PROMPT_LENGTH \
    rollout.response_length=$RESPONSE_LENGTH \
    rollout.tensor_model_parallel_size=$NGPUS \
    rollout.gpu_memory_utilization=0.8


python eval/process_eval.py $SAVE_PATH eval/$MODEL_NAME-ID/accuracy.json

