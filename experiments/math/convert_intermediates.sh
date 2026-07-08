#!/bin/bash
set -e

BASE_DIR="/data/user_data/gyeongwk/checkpoints/math-task"
SCRIPT="/home/gyeongwk/compositional-generality/scripts/model_merger.py"

# Intermediate checkpoints only (highest steps already uploaded)
declare -A CHECKPOINTS=(
    ["math-onpolicy-GRPO-Qwen3-1.7B/global_step_600/actor"]="gyeongwk/Math-On-policy-GRPO-Qwen3-1.7B-step-600"
    ["math-onpolicy-GRPO-Qwen3-1.7B/global_step_1600/actor"]="gyeongwk/Math-On-policy-GRPO-Qwen3-1.7B-step-1600"
    ["math-onpolicy-PosNeg-Qwen3-1.7B/global_step_600/actor"]="gyeongwk/Math-On-policy-PosNeg-Qwen3-1.7B-step-600"
)

for rel_path in "${!CHECKPOINTS[@]}"; do
    local_dir="${BASE_DIR}/${rel_path}"
    hf_repo="${CHECKPOINTS[$rel_path]}"
    echo "========================================"
    echo "Converting: ${local_dir}"
    echo "Uploading to: ${hf_repo}"
    echo "========================================"
    python "${SCRIPT}" --local_dir "${local_dir}" --hf_upload_path "${hf_repo}"
    echo "Done: ${hf_repo}"
done

echo "All intermediate conversions and uploads complete."
