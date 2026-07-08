set -e
set -x
export VLLM_USE_V1=0

BACKBONE_PATH=gyeongwk/Math-On-policy-GRPO-Qwen3-1.7B-step-1800
MATH_DATA_PATH=data/math
TEACHER_DIR=${MATH_DATA_PATH}/teacher/qwen3-1.7b
N_SAMPLES=16
MAX_TOKENS=4096
TEMPERATURE=1.0
PORT=8765

mkdir -p ${TEACHER_DIR}

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

if [ -f ${TEACHER_DIR}/math-easy-train.parquet ]; then
    echo "math-easy already complete, skipping."
else
    python scripts/generation/generate_with_vllm_server.py \
        --data-path ${MATH_DATA_PATH}/math-easy/train.parquet \
        --output-path ${TEACHER_DIR}/math-easy-train.parquet \
        --server-url http://127.0.0.1:${PORT} \
        --n-samples ${N_SAMPLES} \
        --temperature ${TEMPERATURE} \
        --max-tokens ${MAX_TOKENS} \
	--timeout 600 \
        --num-workers 16
fi

kill ${VLLM_PID}
echo "Teacher dataset saved to ${TEACHER_DIR}"
