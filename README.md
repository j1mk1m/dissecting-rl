# Dissecting RL

This repository dissects LLM post-training along two independent axes, **where the training data comes from** (off-policy to on-policy) and **how each sample is weighted in the loss** (positive-only to group-normalized), to isolate which ingredients of RL (GRPO) actually matter for generalization.

Experiments run on two tasks:

- **String task:** compositional string manipulation; train on depth-2 compositions, evaluate on depths 1–8 (Llama-3.1-8B).
- **Math task:** competition-style math split into easy / medium / hard; train on easy (or easy+medium), evaluate on all three (Qwen3-1.7B).

Built on [verl](https://github.com/volcengine/verl) (Volcano Engine RL for LLMs). Based on the [RL-Compositionality](https://github.com/PRIME-RL/RL-Compositionality) codebase ([paper](https://huggingface.co/papers/2509.25123)).

---

## Tasks

### String task

Each problem presents a Python `main_solution` function built by composing primitives from a library of 25 string operators (e.g. `reverse_words`, `sort_chars`, `shift_chars`). The model must predict the output of the function on a given input string without executing code.

```
You are given a code:

def main_solution(s):
    return func_9(func_3(s))  # mirror_str(sort_chars(s))

Can you predict the output of `main_solution("hello")` without writing any code?
Please reason and put your final answer in the following json format: {"output": <your output>}
```

**Difficulty levels** correspond to composition depth (number of operators applied). Models train on level 2 and are evaluated across levels 1–8 to measure compositional generalization.

### Math task

Math problems are bucketed into `math-easy`, `math-medium` and `math-hard` splits (`data/math/<split>/{train,eval}.parquet`). Answers are graded by `verl/utils/reward_score/entropy_math` (a math-verify-style grader). Each configuration has two variants: `*_math.sh` trains on easy only, and `*_math_medium.sh` trains on easy + medium. Both evaluate on easy, medium and hard.

---

## Unified Two-Axis Framework

This project decomposes post-training into two orthogonal axes and varies each independently. Crossing these two axes produces a grid of methods that interpolates between standard off-policy SFT and full on-policy GRPO, isolating the contribution of each component.

|  | **Positive samples only** | **Positive + negative samples** |
|---|---|---|
| **Off-policy** (Teacher / Bootstrap) | SFT | POS+NEG, REINFORCE+Baseline, GRPO on fixed rollouts |
| **On-policy** | On-policy SFT | POS+NEG, REINFORCE+Baseline, GRPO |

### Data axis

The **data axis** controls where training trajectories come from — ranging from fully off-policy to fully on-policy. See [Data sources](#data-sources) under Training for the concrete settings (On-policy, Teacher, Bootstrap).

Key finding: on-policy data alone is not sufficient. Positive-only on-policy SFT contracts response length and entropy and underperforms teacher-distilled off-policy SFT at higher composition levels, because training on only correct short rollouts suppresses the long chain-of-thought trajectories needed for deeper composition.

### Loss function axis

The **loss function axis** controls how the model is updated given a batch of rollouts. The key insight is that loss functions differ in the weights they assign to positive ($r=1$) and negative ($r=0$) samples. All methods with binary reward reduce to:

$$\ell(\theta) = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G w_i \cdot \log \pi_\theta(y_i \mid x)\right]$$

The five loss functions, ordered from positive-only to fully normalized:

**SFT** — trains only on correct rollouts, weighting each by $1/T$ (sequence length normalization). Negative samples are ignored.

$$\ell_{\text{SFT}}(\theta) = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \mathbf{1}[r_i=1] \cdot \frac{1}{T}\log\pi_\theta(y_i \mid x)\right]$$

**GRPOMask** — applies GRPO-style group normalization to positive samples only; negative gradients are masked out. Isolates the effect of group-normalized weighting without negative samples.

$$\ell_{\text{GRPOMask}}(\theta) = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \mathbf{1}[r_i=1] \cdot \frac{r_i-\mu}{\sigma}\cdot\log\pi_\theta(y_i \mid x)\right]$$

**POS+NEG** — naively adds negative gradients: weight $+1$ for correct rollouts, $-1$ for incorrect. Equivalent to REINFORCE with rewards in $\{+1, -1\}$.

$$\ell_{\text{POS+NEG}}(\theta) = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \bigl(\mathbf{1}[r_i=1] - \mathbf{1}[r_i=0]\bigr)\cdot\log\pi_\theta(y_i \mid x)\right]$$

**REINFORCE+Baseline** — adds a group-mean baseline $\mu$ to balance gradient magnitudes across tasks of varying difficulty. Weight is $(1-\mu)$ for positives and $-\mu$ for negatives.

$$\ell_{\text{REINFORCE+}}(\theta) = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G (r_i - \mu)\cdot\log\pi_\theta(y_i \mid x)\right]$$

**GRPO** — further normalizes by the within-group standard deviation $\sigma$. Weight is $(1-\mu)/\sigma$ for positives and $-\mu/\sigma$ for negatives.

$$\ell_{\text{GRPO}}(\theta) = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{r_i-\mu}{\sigma}\cdot\log\pi_\theta(y_i \mid x)\right]$$

Here $\mu$ and $\sigma$ are the mean and standard deviation of rewards within each rollout group of size $G$.

**Summary table** (config flags shown for reference):

| Loss | $w^+$ | $w^-$ | `reward_baseline` | Extra flags |
|---|---|---|---|---|
| SFT | $1/T$ | $0$ | `"none"` | `loss_agg_mode=seq-mean-token-mean` |
| GRPOMask | $(1-\mu)/\sigma$ | $0$ | `"grpo_mask"` | — |
| POS+NEG | $1$ | $-1$ | `"none"` | `enable_negative_sample_training: true` |
| REINFORCE+Baseline | $1-\mu$ | $-\mu$ | `"mean"` | — |
| GRPO | $(1-\mu)/\sigma$ | $-\mu/\sigma$ | `"mean"` | `reward_normalize_std: true` |

Key findings on the loss axis: introducing negative gradients (POS+NEG) recovers a large fraction of the gap to GRPO. Adding the group-mean baseline (REINFORCE+Baseline) closes nearly all of the remaining gap. The additional $\sigma$ normalization in full GRPO contributes relatively little.

---

## Setup

```bash
git clone <anonymized-repo-url> dissecting-rl
cd dissecting-rl
pip install -e ".[vllm]"
```

Model checkpoints are referenced as `anonymous/<model>` for review. Shell scripts read the Hugging Face namespace from `HF_USER` (default `anonymous`), so set `export HF_USER=<your-hf-namespace>` to use your own uploads. `scripts/analysis/fetch_wandb_training_dynamics.py` needs `WANDB_ENTITY`.

**Requirements:** 4× A100 GPUs (all scripts use `CUDA_VISIBLE_DEVICES=0,1,2,3`).

---

## Data

`data/` is gitignored and must be populated locally.

**String task** (`data/string_task/`)

| Path | Description |
|---|---|
| `stage2_level2/train.parquet` | Train set, depth-2 compositions |
| `stage2_level1to8/test.parquet` | Eval set, depths 1–8 |
| `teacher-grpo/rollout.parquet` | Rollouts from the on-policy GRPO model (`anonymous/On-policy-GRPO`) |
| `teacher-bootstrap/rollout.parquet` | Rollouts from the initial model `anonymous/stage1-rft` |

Regenerate the string datasets with:
```bash
python scripts/data_preprocess/string_data.py
python scripts/data_preprocess/string_data_analysis.py --input <parquet>
```

**Math task** (`data/math/`)

| Path | Description |
|---|---|
| `math-{easy,medium,hard}/{train,eval}.parquet` | Difficulty splits |
| `teacher/qwen3-1.7b/math-{easy,medium}-train.parquet` | Rollouts from `anonymous/Math-On-policy-GRPO-Qwen3-1.7B-step-1800` |
| `bootstrap/qwen3-1.7b/math-{easy,medium}-train.parquet` | Rollouts from base `Qwen/Qwen3-1.7B` |

Check that the math parquets are compatible with the training pipeline:
```bash
python scripts/test_math_dataset.py
```

---

## Training

All methods use the OSFT recipe (`recipe.osft.main_osft`). The loss function is selected by the flags in the summary table above, and the data source by `trainer.data_source.mode`.

### Data sources

**On-policy** (`mode=on_policy`): the policy samples $G$ completions per prompt at every step.

**Teacher** (`mode=teacher`): trains on a fixed parquet of rollouts from a stronger model, the GRPO-trained checkpoint. The student is never used to generate rollouts.

**Bootstrap** (`mode=teacher`, base-model rollouts): same mechanism as Teacher, but the fixed rollouts come from the *initial* model. The data is on-policy at step 0 and grows increasingly off-policy as training proceeds.

`mode=iterative` (regenerate rollouts from the current checkpoint every `trainer.data_source.iterative_k` steps) and `mode=bootstrap` are also implemented in `recipe/osft/data_source_controller.py`, but the scripts in this repo don't use them.

### String task scripts

Backbone: `anonymous/stage1-rft` (Llama-3.1-8B, RFT on depth-1 data). W&B project: `string-task`.

```bash
# {sft, grpo, pos_neg, reinforce_baseline} for each data source
bash experiments/string/data-onpolicy-loss-fn-vary/onpolicy_grpo.sh
bash experiments/string/data-teacher-loss-fn-vary/teacher_grpo.sh
bash experiments/string/data-bootstrap-loss-fn-vary/bootstrap_grpo.sh
```

`experiments/string/stage1_rft.sbatch` produces the stage-1 backbone. `experiments/string/run_modal.py` runs any script on Modal.

### Math task scripts

Backbone: `Qwen/Qwen3-1.7B`. W&B project: `math-task`. All scripts live in `experiments/math/` and are named `{onpolicy,teacher,bootstrap}_{sft,grpo,pos_neg,reinforce_baseline}_math[_medium].sh`.

```bash
# 1. Pre-generate off-policy rollouts (starts a local vLLM server, TP=4)
bash experiments/math/generate_bootstrap.sh         # base-model rollouts, easy
bash experiments/math/generate_teacher.sh           # GRPO-teacher rollouts, easy
#    (*_medium.sh variants add the medium split)

# 2. Train
bash experiments/math/onpolicy_grpo_math.sh         # train on easy
bash experiments/math/teacher_sft_math_medium.sh    # train on easy + medium

# 3. Convert FSDP checkpoints to HF format and upload
bash experiments/math/convert_and_upload.sh
```

### Key hyperparameters

| Parameter | String task | Math task |
|---|---|---|
| Backbone | `anonymous/stage1-rft` (Llama-3.1-8B) | `Qwen/Qwen3-1.7B` |
| GPUs | 4 | 4 |
| Rollouts per prompt ($G$) | 16 | 16 |
| Train batch size | 16 prompts | 16 prompts |
| Max prompt / response length | 1024 / 4096 | 1024 / 4096 |
| Learning rate | 1e-6, 5 warmup steps | 1e-6, 5 warmup steps |
| Epochs | 1 | 1 |
| Validation frequency | 25 (on-policy) / 100 (off-policy) steps | 50 steps |

### Cluster (Slurm)

`deploy/launcher.py` submits any experiment script through a Slurm template:

```bash
python deploy/launcher.py experiments/math/onpolicy_grpo_math.sh
python deploy/launcher.py experiments/... --template deploy/template_A100_80GB.sbatch
```

| Template | Notes |
|---|---|
| `template.sbatch` | Default (`general` partition) |
| `template_A100_80GB.sbatch` | Requests A100 80GB |
| `template_light.sbatch` | Lighter resource request |
| `template_cpu.sbatch` | CPU-only jobs |
| `template_preempt.sbatch` | Preemptible jobs |
| `template_modal.sbatch` | Modal cloud |

### Resuming from checkpoint

```bash
python3 -m recipe.osft.main_osft \
    ... \
    trainer.resume_mode=auto \
    trainer.default_local_dir=<checkpoint_dir>
```

---

## Offline Generation (string task)

String-task teacher and bootstrap rollouts are generated with a separate vLLM server and client:

```bash
bash experiments/util/serve_vllm.sh          # or: sbatch deploy/serve_vllm.sbatch
bash experiments/util/generation_client.sh   # or: sbatch deploy/generation_client.sbatch
```

Both tasks use `scripts/generation/generate_with_vllm_server.py`. It streams requests with retry/backoff and checkpoints progress to a `.jsonl` file, so interrupted runs resume where they left off. Key flags: `--data-path`, `--output-path`, `--n-samples`, `--checkpoint-path`, `--num-workers`.

---

## Evaluation & Analysis

During training, validation runs every `trainer.test_freq` steps with greedy decoding and logs per-split accuracy to W&B (`val/...`).

- `scripts/evaluation/process_eval.py`: scores a rollout parquet and reports per-level accuracy broken down by operator.
- `scripts/analysis/fetch_wandb_training_dynamics.py`: pulls training curves from W&B into `results/*.csv`.
- `scripts/analysis/plot_training_dynamics.py` and the other `plot_*.py` scripts: paper figures written to `reports/imgs/`.

`results/` and `reports/` (paper draft, figures, write-ups) are gitignored and kept locally only.

---

## Repository Structure

```
verl/                           # verl framework (Ray, FSDP, vLLM rollout)
recipe/osft/                    # Training recipe
  main_osft.py                  # Entry point
  osft_trainer.py               # RayOSFTTrainer training loop
  osft_sample_selection.py      # Per-sample weight computation (all loss variants)
  dp_actor.py                   # Weighted NLL loss + backward
  data_source_controller.py     # Trajectory source selection
  config/osft_trainer.yaml      # All configurable options
recipe/dpo/                     # DPO trainer (not used in the main experiments)

experiments/
  string/                       # String task: {onpolicy,teacher,bootstrap} × loss
  math/                         # Math task: generation, training, checkpoint conversion
  util/                         # vLLM server + generation client
  _archive/                     # Old DPO / GRPO / SFT templates
deploy/                         # Slurm launcher and templates

scripts/
  data_preprocess/              # String dataset generation and analysis
  generation/                   # Offline rollout generation via vLLM server
  evaluation/                   # Rollout scoring
  analysis/                     # W&B fetching, plotting, rollout analysis
  model_merger.py               # FSDP checkpoint → HF conversion
  test_math_dataset.py          # Math parquet sanity checks
```

---

## Monitoring

Runs log to W&B under the projects `string-task` and `math-task`. Key metrics:

| Metric | Description |
|---|---|
| `reward/score/mean` | Mean rollout reward before filtering |
| `training/n_positive_seq` / `n_negative_seq` | Positive / negative samples per batch |
| `training/weight_mean` | Mean sample weight sent to the actor |
| `training/data_source_is_off_policy` | 1 when training on teacher/bootstrap data |
| `actor/pg_loss`, `actor/grad_norm` | Loss and gradient norm |
| `rollout/avg_response_length` | Mean response length |
| `val/<split>/...` | Validation accuracy and entropy per level / difficulty split |
