#!/bin/bash
set -e

BASE_DIR="/data/user_data/gyeongwk/checkpoints/math-task/math-onpolicy-Reinforce-Baseline-Qwen3-1.7B"

for step_dir in "${BASE_DIR}"/global_step_*/actor; do
    step=$(basename "$(dirname "${step_dir}")" | sed 's/global_step_//')
    hf_repo="gyeongwk/Math-On-policy-Reinforce-Baseline-Qwen3-1.7B-step-${step}"
    echo "Converting step ${step}: ${step_dir} -> ${hf_repo}"
    python scripts/model_merger.py --local_dir "${step_dir}" --hf_upload_path "${hf_repo}"
done
