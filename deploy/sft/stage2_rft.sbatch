#!/bin/bash
#SBATCH --partition general 
#SBATCH --qos normal 
#SBATCH --time 1-23:00 # set time
#SBATCH --job-name=stage2_rft
#SBATCH --chdir=/home/gyeongwk/RL-Compositionality # working directory

#SBATCH --gres=gpu:A100_80GB:4 
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
EXPERIMENT_NAME=stage2-rft-with-code

NNODES=1
NGPUS=4
DATA_PATH=data/string_task/level2_with_code/train.parquet
N_SAMPLES=16
MODEL_PATH=gyeongwk/stage1-rft
TEMPERATURE=1.0
PROMPT_LENGTH=1024
RESPONSE_LENGTH=4096
BASE_SAVE_PATH=data/string_task/$EXPERIMENT_NAME
SAVE_PATH=$BASE_SAVE_PATH/rollout.parquet
RFT_DATA_SAVE_PATH=$BASE_SAVE_PATH/rft_data
BATCH_SIZE=256

# Run generation with MODEL_PATH model
echo "Running generation"
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
    rollout.gpu_memory_utilization=0.8

# Convert to SFT format
echo "Converting to SFT dataset"
python3 examples/data_preprocess/string_manipulation_sft.py \
    --gen_path $SAVE_PATH \
    --data_path $DATA_PATH \
    --save_path ${RFT_DATA_SAVE_PATH} \
    --val_size 16 \
    --max_correct_ratio 1.1 \
    --n_samples 1

# Run SFT on MODEL_PATH model
MODEL_PATH=gyeongwk/stage1-rft
NNODES=1
SP_SIZE=1
MAX_LENGTH=5120
TRAIN_FILES=$BASE_SAVE_PATH/rft_data/train.parquet 
VAL_FILES=$BASE_SAVE_PATH/rft_data/test.parquet 
BATCH_SIZE=128
EPOCHS=1
SAVE_DIR=/data/user_data/gyeongwk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

echo "Running SFT"

torchrun --standalone --nnodes=${NNODES} --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=${MAX_LENGTH} \
    data.truncation=right \
    optim.lr=2e-5 \
    data.train_batch_size=${BATCH_SIZE} \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=${MODEL_PATH} \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=${EPOCHS} \
    ulysses_sequence_parallel_size=${SP_SIZE} \
    use_remove_padding=true

N_SAMPLES=1
DATA_PATH=data/string_task/stage2_level1to8/test.parquet
TEMPERATURE=0.0
PROMPT_LENGTH=1024
RESPONSE_LENGTH=8192
BATCH_SIZE=16

MODEL_PATH=/data/user_data/gyeongwk/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME
SAVE_PATH=eval/$EXPERIMENT_NAME/rollout.parquet

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

python eval/process_eval.py $SAVE_PATH eval/$EXPERIMENT_NAME/accuracy.json

