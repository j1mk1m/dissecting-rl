#!/bin/bash
#SBATCH --partition general
#SBATCH --time 1-23:00 # set time
#SBATCH --job-name=vllm
#SBATCH --chdir=/home/gyeongwk # working directory

#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=/home/gyeongwk/logs/%x-%j.out # output log
#SBATCH --error=/home/gyeongwk/logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

export NCCL_P2P_DISABLE=1
export VLLM_CACHE_DIR=/data/user_data/$USER/.cache/vllm
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

source ~/.bashrc
conda activate rl_comp


MODEL="gyeongwk/stage2-osft"
PORT=8084

if ss -tulwn | grep -q ":$PORT "; then
    echo "Port $PORT is already in use. Exiting..."
    exit 1
else
    vllm serve $MODEL --host 0.0.0.0 --port $PORT --tensor-parallel-size 4 --max-model-len 8192 --rope-scaling '{"rope_type":"dynamic","factor":4}'
fi