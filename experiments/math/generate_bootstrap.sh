set -e
set -x
export VLLM_USE_V1=0

BACKBONE_PATH=Qwen/Qwen3-1.7B
MATH_DATA_PATH=data/math
BOOTSTRAP_DIR=${MATH_DATA_PATH}/bootstrap/qwen3-1.7b
N_SAMPLES=16
MAX_TOKENS=4096
MAX_MODEL_LEN=6144
TEMPERATURE=1.0
PORT=$(( 20000 + (${SLURM_JOB_ID:-$$} % 20000) ))

mkdir -p ${BOOTSTRAP_DIR}

# Start vLLM server in background
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model ${BACKBONE_PATH} \
    --port ${PORT} \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len ${MAX_MODEL_LEN} \
    --dtype bfloat16 &
VLLM_PID=$!

echo "Waiting for vLLM server to start (pid=${VLLM_PID}, port=${PORT})..."
until curl -s http://127.0.0.1:${PORT}/v1/models > /dev/null 2>&1; do
    sleep 5
done

SERVED_MODEL=$(curl -s http://127.0.0.1:${PORT}/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
if [ "${SERVED_MODEL}" != "${BACKBONE_PATH}" ]; then
    echo "ERROR: port ${PORT} is serving '${SERVED_MODEL}', not '${BACKBONE_PATH}'. Another process is squatting on this port. Aborting." >&2
    kill ${VLLM_PID} 2>/dev/null
    exit 1
fi
echo "vLLM server ready, confirmed serving ${BACKBONE_PATH}."

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
        --max-model-len ${MAX_MODEL_LEN} \
        --tokenizer-path ${BACKBONE_PATH} \
	--timeout 600 \
        --num-workers 16
fi

kill ${VLLM_PID}
echo "Bootstrap dataset saved to ${BOOTSTRAP_DIR}"
