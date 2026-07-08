#!/bin/bash
set -e

BASE_DIR="/data/user_data/gyeongwk/checkpoints/math-task"
SCRIPT="/home/gyeongwk/compositional-generality/scripts/model_merger.py"

declare -A HF_PREFIX=(
    ["math-onpolicy-GRPO-Qwen3-1.7B"]="gyeongwk/Math-On-policy-GRPO-Qwen3-1.7B"
    ["math-onpolicy-PosNeg-Qwen3-1.7B"]="gyeongwk/Math-On-policy-PosNeg-Qwen3-1.7B"
    ["math-onpolicy-Reinforce-Baseline-Qwen3-1.7B"]="gyeongwk/Math-On-policy-Reinforce-Baseline-Qwen3-1.7B"
    ["math-onpolicy-SFT-Qwen3-1.7B"]="gyeongwk/Math-On-policy-SFT-Qwen3-1.7B"
)

for exp in "${!HF_PREFIX[@]}"; do
    hf_base="${HF_PREFIX[$exp]}"
    for step_dir in "${BASE_DIR}/${exp}"/global_step_*/actor; do
        [ -d "${step_dir}" ] || continue
        step=$(basename "$(dirname "${step_dir}")" | sed 's/global_step_//')
        hf_repo="${hf_base}-step-${step}"
        echo "========================================"
        echo "Converting: ${step_dir}"
        echo "Uploading to: ${hf_repo}"
        echo "========================================"
        python "${SCRIPT}" --local_dir "${step_dir}" --hf_upload_path "${hf_repo}"
        echo "Done: ${hf_repo}"
    done
done

echo "All conversions and uploads complete."
