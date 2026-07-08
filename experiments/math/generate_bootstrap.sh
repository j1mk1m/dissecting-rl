set -e
set -x
export VLLM_USE_V1=0

BACKBONE_PATH=Qwen/Qwen3-1.7B
MATH_DATA_PATH=data/math
BOOTSTRAP_DIR=${MATH_DATA_PATH}/bootstrap/qwen3-1.7b
N_SAMPLES=16
MAX_TOKENS=4096
TEMPERATURE=1.0
PORT=8765

mkdir -p ${BOOTSTRAP_DIR}

# Start vLLM server in background
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model ${BACKBONE_PATH} \
    --port ${PORT} \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 6144 \
    --dtype bfloat16 &
VLLM_PID=$!

echo "Waiting for vLLM server to start (pid=${VLLM_PID})..."
until curl -s http://127.0.0.1:${PORT}/v1/models > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM server ready."

if [ -f ${BOOTSTRAP_DIR}/math-easy-train.parquet ]; then
    echo "math-easy already complete, skipping."
else
    python scripts/generation/generate_with_vllm_server.py \
        --data-path ${MATH_DATA_PATH}/math-easy/train.parquet \
        --output-path ${BOOTSTRAP_DIR}/math-easy-train.parquet \
        --server-url http://127.0.0.1:${PORT} \
        --n-samples ${N_SAMPLES} \
        --temperature ${TEMPERATURE} \
        --max-tokens ${MAX_TOKENS} \
	--timeout 600 \
        --num-workers 16
fi

kill ${VLLM_PID}
echo "Bootstrap dataset saved to ${BOOTSTRAP_DIR}"
