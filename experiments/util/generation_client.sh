MODEL="gyeongwk/stage2-osft"
MODEL_NAME="sft-onpolicy"
MACHINE="babel-x9-16"
PORT="8084"

python3 scripts/generation/generate_with_vllm_server.py \
  --data-path data/string_task/stage2_level1to8/test.parquet \
  --output-path eval/$MODEL_NAME/rollout.parquet \
  --prompt-key prompt \
  --response-key responses \
  --server-url http://$MACHINE:$PORT \
  --model $MODEL \
  --n-samples 1 \
  --temperature 0 \
  --top-p 1.0 \
  --max-tokens 4096 \
  --num-workers 4

python scripts/evaluation/process_eval.py eval/$MODEL_NAME/rollout.parquet eval/$MODEL_NAME/accuracy.json