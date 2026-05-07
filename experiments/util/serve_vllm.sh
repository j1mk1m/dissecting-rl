export NCCL_P2P_DISABLE=1
export VLLM_CACHE_DIR=/data/user_data/$USER/.cache/vllm
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

MODEL="gyeongwk/On-policy-GRPO"
PORT=8084

if ss -tulwn | grep -q ":$PORT "; then
    echo "Port $PORT is already in use. Exiting..."
    exit 1
else
    vllm serve $MODEL --host 0.0.0.0 --port $PORT --tensor-parallel-size 4 
fi
