#!/bin/bash
set -e

BASE_DIR="checkpoints/math-task/math-onpolicy-GRPO-Qwen3-1.7B"

for step_dir in "${BASE_DIR}"/global_step_*/actor; do
    step=$(basename "$(dirname "${step_dir}")" | sed 's/global_step_//')
    hf_repo="${HF_USER:-anonymous}/Math-On-policy-GRPO-Qwen3-1.7B-step-${step}"
    echo "Converting step ${step}: ${step_dir} -> ${hf_repo}"
    python scripts/model_merger.py --local_dir "${step_dir}" --hf_upload_path "${hf_repo}"
done
