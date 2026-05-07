MODEL="gyeongwk/On-policy-GRPO"
MODEL_NAME="teacher-grpo"
MACHINE="babel-o9-16"
PORT="8084"

python3 scripts/generation/generate_with_vllm_server.py \
  --data-path data/string_task/stage2_level2/train.parquet \
  --output-path data/string_task/$MODEL_NAME/rollout.parquet \
  --prompt-key prompt \
  --response-key responses \
  --server-url http://$MACHINE:$PORT \
  --model $MODEL \
  --n-samples 16 \
  --temperature 1.0 \
  --max-tokens 4096 \
  --num-workers 4 \
  --checkpoint-path /data/user_data/gyeongwk/checkpoints/$MODEL_NAME-rollout.checkpoint.jsonl

#python scripts/evaluation/process_eval.py eval/$MODEL_NAME/rollout.parquet eval/$MODEL_NAME/accuracy.json
