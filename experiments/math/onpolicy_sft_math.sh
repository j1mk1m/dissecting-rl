set -e
set -x
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
NGPUS=4
VISIBLE_DEVICES="0,1,2,3"

AFS_PATH=${AFS_PATH:-.}
MATH_DATA_PATH=data/math
TRAIN_FILES="[$MATH_DATA_PATH/math-easy/train.parquet,$MATH_DATA_PATH/math-medium/train.parquet]"
VAL_FILES="[$MATH_DATA_PATH/math-easy/eval.parquet,$MATH_DATA_PATH/math-medium/eval.parquet,$MATH_DATA_PATH/math-hard/eval.parquet]"

LR=1e-6
BACKBONE_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
MAX_PROMPT_LENGTH=1024
MAX_GEN_LENGTH=4096
ROLLOUT_N=16
EXPERIMENT="math-onpolicy-SFT"

PROJECT_NAME="math-task"

OUTPUT_DIR="${AFS_PATH}/checkpoints/${PROJECT_NAME}/${EXPERIMENT}"


CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} \
python3 -m recipe.osft.main_osft \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.shuffle=False \
    data.train_batch_size=16 \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_GEN_LENGTH} \
    actor_rollout_ref.model.path=${BACKBONE_PATH} \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.enable_train_temperature=False \
    trainer.enable_negative_sample_training=False \
    trainer.reward_std_eps=1e-8 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT} \
    trainer.val_before_train=True \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.default_hdfs_dir=null \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
    trainer.validation_data_dir=${OUTPUT_DIR}/rollout_eval_data \
    trainer.test_freq=50 \
    +trainer.log_freq=1 \
    trainer.total_epochs=1
