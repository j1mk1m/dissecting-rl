MODEL="gyeongwk/On-policy-GRPO"
MODEL_NAME="teacher-grpo"
MACHINE="babel-w5-28"
PORT="8082"

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
  --num-workers 4

#python scripts/evaluation/process_eval.py eval/$MODEL_NAME/rollout.parquet eval/$MODEL_NAME/accuracy.json
