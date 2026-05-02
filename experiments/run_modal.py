"""
Generic Modal launcher for experiment shell scripts.

This avoids creating one Python Modal file per experiment config:
- Keep experiment configuration in your existing `.sh` scripts.
- Run the same script on Modal GPUs through this single launcher.

Usage:
  modal run experiments/data-onpolicy-loss-fn-vary/onpolicy_grpo_modal.py
  modal run experiments/data-onpolicy-loss-fn-vary/onpolicy_grpo_modal.py \
      --script-path experiments/data-onpolicy-loss-fn-vary/onpolicy_pos_neg.sh

Optional:
  MODAL_GPU="A100-80GB:8" modal run experiments/data-onpolicy-loss-fn-vary/onpolicy_grpo_modal.py \
      --script-path experiments/data-onpolicy-loss-fn-vary/onpolicy_grpo.sh
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import List

import modal


APP_NAME = "string-task-experiment-runner"
REPO_ROOT = "/root/compositional-generality"
LOCAL_REPO_ROOT = str(Path(__file__).resolve().parents[1])
DEFAULT_SCRIPT_PATH = "experiments/data-onpolicy-loss-fn-vary/onpolicy_grpo.sh"

# Make sure these volumes exist and contain your data before running:
# - `cg-string-task-data` mounted at /root/RL-Compositionality/data
# - `cg-user-data` mounted at /data/user_data/gyeongwk
CHECKPOINTS_VOLUME = modal.Volume.from_name("cg-string-task-checkpoints", create_if_missing=True)

# Use a prebuilt training image if provided, otherwise use a CUDA-ready base.
BASE_IMAGE = os.environ.get(
    "MODAL_TRAINING_IMAGE",
    "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
)

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.10")
    .apt_install("git", "build-essential")
    .pip_install("pip>=24.0", "setuptools", "wheel")
    .add_local_dir(LOCAL_REPO_ROOT, remote_path=REPO_ROOT)
)

app = modal.App(APP_NAME)


def _validate_script_path(script_path: str) -> str:
    script_abs = (Path(REPO_ROOT) / script_path).resolve()
    repo_root_abs = Path(REPO_ROOT).resolve()
    if repo_root_abs not in script_abs.parents and script_abs != repo_root_abs:
        raise ValueError(f"Script path must stay within repo root: {script_path}")
    if not script_abs.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    return str(script_abs)


@app.function(
    image=image,
    gpu=os.environ.get("MODAL_GPU", "A100-80GB:4"),
    cpu=16,
    memory=65536,
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        "/root/compositional-generality/checkpoints": CHECKPOINTS_VOLUME,
    },
)
def run_experiment_on_modal(
    script_path: str = DEFAULT_SCRIPT_PATH,
    script_args: str = "",
    extra_env: str = "",
    install_project: bool = True,
) -> None:
    os.chdir(REPO_ROOT)

    if install_project:
        subprocess.run(
            ["python3", "-m", "pip", "install", "-e", ".[vllm]"],
            check=True,
        )
        subprocess.run(
            ["python3", "-m", "pip", "install", "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"],
            check=True,
        )
        subprocess.run(
            ["python3", "-m", "pip", "install", "transformers==4.51.1"],
        )

    # Keep base env in sync with local scripts.
    os.environ["HYDRA_FULL_ERROR"] = os.environ.get("HYDRA_FULL_ERROR", "1")

    script_abs = _validate_script_path(script_path)

    if extra_env.strip():
        for item in shlex.split(extra_env):
            if "=" not in item:
                raise ValueError(f"Invalid extra_env item (expected KEY=VALUE): {item}")
            key, value = item.split("=", 1)
            os.environ[key] = value

    command: List[str] = ["bash", script_abs]
    if script_args.strip():
        command.extend(shlex.split(script_args))

    print("Running script command:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)

    # Persist checkpoints and generated rollout files to Modal Volume.
    CHECKPOINTS_VOLUME.commit()


@app.local_entrypoint()
def main(
    script_path: str = DEFAULT_SCRIPT_PATH,
    script_args: str = "",
    extra_env: str = "",
    install_project: bool = True,
):
    run_experiment_on_modal.remote(
        script_path=script_path,
        script_args=script_args,
        extra_env=extra_env,
        install_project=install_project,
    )
