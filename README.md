# Compositional Generality

This repository studies compositional generalization in LLMs using a string manipulation task. Models are trained on compositions of a fixed set of string operators at a given depth and evaluated on held-out depths to measure how well learned skills compose.

Built on [verl](https://github.com/volcengine/verl) (Volcano Engine RL for LLMs). Based on the [RL-Compositionality](https://github.com/PRIME-RL/RL-Compositionality) codebase ([paper](https://huggingface.co/papers/2509.25123)).

---

## Task

Each problem presents a Python `main_solution` function built by composing primitives from a library of 25 string operators (e.g. `reverse_words`, `sort_chars`, `shift_chars`). The model must predict the output of the function on a given input string without executing code.

```
You are given a code:

def main_solution(s):
    return func_9(func_3(s))  # mirror_str(sort_chars(s))

Can you predict the output of `main_solution("hello")` without writing any code?
Please reason and put your final answer in the following json format: {"output": <your output>}
```

**Difficulty levels** correspond to composition depth (number of operators applied). Models train on level 2 and are evaluated across levels 1–8 to measure compositional generalization.

---

## Unified Two-Axis Framework

This project decomposes post-training into two orthogonal axes and varies each independently. Crossing these two axes produces a grid of methods that interpolates between standard off-policy SFT and full on-policy GRPO, isolating the contribution of each component.

|  | **Positive samples only** | **Positive + negative samples** |
|---|---|---|
| **Off-policy** | SFT | DPO |
| **On-policy** | On-policy SFT | GRPO |

### Data axis

The **data axis** controls where training trajectories come from — ranging from fully off-policy to fully on-policy. See the [Data source modes](#data-source-modes) section under Training for the four concrete settings (Teacher, Bootstrap, Iterative, On-policy).

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
git clone <this-repo>
cd compositional-generality
pip install -e ".[vllm]"
conda activate osft
```

**Requirements:** 4× A100 GPUs (standard scripts use `CUDA_VISIBLE_DEVICES=0,1,2,3`).

---

## Data

| Path | Description |
|---|---|
| `data/string_task/stage2_level2/train.parquet` | Train set, depth-2 compositions |
| `data/string_task/stage2_level1to8/test.parquet` | Eval set, depths 1–8 |
| `data/string_task/stage2_level2_rft/` | Level-2 data for RFT stage |
| `data/string_task/teacher-grpo/rollout.parquet` | Rollouts from GRPO teacher model |
| `data/string_task/teacher-bootstrap/rollout.parquet` | Rollouts from base model (bootstrap) |
| `data/string_task/teacher-rl-checkpoint/` | Rollouts from RL checkpoint teacher |

To regenerate datasets:
```bash
python scripts/data_preprocess/string_data.py
python scripts/data_preprocess/string_data_analysis.py --input <parquet>
```

---

## Training

All methods use the OSFT recipe (`recipe.osft.main_osft`) with backbone `gyeongwk/stage1-rft` (Llama-3.1-8B fine-tuned on depth-1 data via RFT).

### Data source modes

The `trainer.data_source.mode` flag selects how training trajectories are sourced. Each data source can be combined with any of the five loss functions above.

**`on_policy`** — The policy rolls out $G$ completions per prompt at every training step. The actor weights are always up to date with the rollout policy. This is the standard online RL regime.

**`bootstrap`** — Rollouts are generated once from the initial base model (before any training) and cached. The same fixed set of trajectories is replayed throughout training — no further rollouts are generated. Since the actor drifts away from the data-generating policy over time this becomes increasingly off-policy.

**`iterative`** — A middle ground: rollouts are generated from the current checkpoint every $k$ training steps (`trainer.data_source.iterative_k`). Between regenerations the cached batch is replayed, introducing a small amount of off-policy lag that grows with $k$.

**`teacher`** — Rollouts come from a separate, fixed teacher model (e.g. `gyeongwk/On-policy-GRPO` trained via GRPO). The teacher trajectories are loaded from a pre-generated parquet file (`data/string_task/teacher-grpo/rollout.parquet`). The student policy is never used for rollouts — training is entirely off-policy relative to the student.

### Quick start

Three experiment groups vary one axis each while holding the others fixed:

```bash
# On-policy rollouts — vary loss function
bash experiments/data-onpolicy-loss-fn-vary/onpolicy_sft.sh
bash experiments/data-onpolicy-loss-fn-vary/onpolicy_grpo.sh
bash experiments/data-onpolicy-loss-fn-vary/onpolicy_pos_neg.sh
bash experiments/data-onpolicy-loss-fn-vary/onpolicy_reinforce_baseline.sh

# Teacher rollouts (gyeongwk/On-policy-GRPO) — vary loss function
bash experiments/data-teacher-loss-fn-vary/teacher_sft.sh
bash experiments/data-teacher-loss-fn-vary/teacher_grpo.sh
bash experiments/data-teacher-loss-fn-vary/teacher_pos_neg.sh
bash experiments/data-teacher-loss-fn-vary/teacher_reinforce_baseline.sh

# Bootstrap rollouts (base model) — vary loss function
bash experiments/data-bootstrap-loss-fn-vary/bootstrap_sft.sh
bash experiments/data-bootstrap-loss-fn-vary/bootstrap_grpo.sh
bash experiments/data-bootstrap-loss-fn-vary/bootstrap_pos_neg.sh
bash experiments/data-bootstrap-loss-fn-vary/bootstrap_reinforce_baseline.sh
```

### Cluster (Slurm)

Use `deploy/launcher.py` to submit any experiment script via a Slurm template:

```bash
# Default template (general partition, 4 GPUs)
python deploy/launcher.py experiments/data-onpolicy-loss-fn-vary/onpolicy_sft.sh

# Specific templates
python deploy/launcher.py experiments/... --template deploy/template_A100_80GB.sbatch
python deploy/launcher.py experiments/... --template deploy/template_flame.sbatch
```

Available templates:

| File | Partition | Notes |
|---|---|---|
| `template.sbatch` | `general` | Default |
| `template_A100_80GB.sbatch` | `general` | Requests A100 80GB explicitly |
| `template_flame.sbatch` | `flame-earlybirds` | Flame cluster early-bird queue |
| `template_light.sbatch` | — | Lighter resource request |
| `template_preempt.sbatch` | — | Preemptible jobs |
| `template_modal.sbatch` | — | Modal cloud |

### Key hyperparameters

| Parameter | Value |
|---|---|
| Backbone | `gyeongwk/stage1-rft` (Llama-3.1-8B) |
| GPUs | 4 |
| Rollouts per prompt ($G$) | 16 |
| Train batch size | 16 prompts |
| Max prompt length | 1024 tokens |
| Max response length | 4096 tokens |
| Learning rate | 1e-6 with 5 warmup steps |
| Epochs | 1 |
| Validation frequency | every 25 steps (on-policy) / 100 steps (off-policy) |

### Resuming from checkpoint

```bash
python3 -m recipe.osft.main_osft \
    ... \
    trainer.resume_mode=auto \
    trainer.default_local_dir=<checkpoint_dir>
```

---

## Offline Generation Pipeline

Teacher and bootstrap rollouts are pre-generated via a two-step vLLM server workflow before running off-policy training scripts.

**Step 1 — start a vLLM server:**
```bash
# Local
bash experiments/util/serve_vllm.sh

# Slurm
sbatch deploy/serve_vllm.sbatch
```

`serve_vllm.sh` checks that the target port is free, then launches `vllm serve` with tensor-parallel-size 4.

**Step 2 — generate rollouts:**
```bash
# Local (edit MODEL, MACHINE, PORT, output path first)
bash experiments/util/generation_client.sh

# Slurm
sbatch deploy/generation_client.sbatch
```

`generation_client.sh` calls `scripts/generation/generate_with_vllm_server.py`, which streams requests to the running server with retry/backoff and checkpoints progress to a `.jsonl` file so interrupted runs resume automatically.

Key flags for the generation script:

| Flag | Description |
|---|---|
| `--data-path` | Input parquet file |
| `--output-path` | Output parquet with generated responses |
| `--n-samples` | Rollouts per prompt (default 16) |
| `--checkpoint-path` | JSONL resume checkpoint |
| `--num-workers` | Concurrent HTTP workers |

---

## Evaluation

```bash
# Generate rollouts from a checkpoint
bash experiments/eval/eval.sh       # greedy decode, levels 1–8

# Score rollouts
python scripts/evaluation/process_eval.py eval/<run>/rollout.parquet eval/<run>/accuracy.json
```

`eval.sh` calls `verl.trainer.main_generation` (4 GPUs, greedy, max 4096 tokens) then pipes the output parquet to `process_eval.py`, which computes per-level accuracy broken down by operator.

---

## Repository Structure

```
verl/                         # verl framework (Ray, FSDP, vLLM rollout)
recipe/osft/                  # Training recipe
  main_osft.py                # Entry point
  osft_trainer.py             # RayOSFTTrainer training loop
  osft_sample_selection.py    # Per-sample weight computation (all loss variants)
  dp_actor.py                 # Weighted NLL loss + backward
  data_source_controller.py   # Trajectory source selection (on-policy/teacher/bootstrap/iterative)
  config/osft_trainer.yaml    # All configurable options

scripts/
  data_preprocess/
    string_data.py            # Dataset generation (25 operators, compositions)
    string_data_analysis.py   # Composition distribution analysis
  generation/
    generate_with_vllm_server.py  # Offline rollout generation via vLLM server
  evaluation/
    process_eval.py           # Score rollout parquets, compute per-level accuracy
  analysis/                   # Plotting and token-length analysis scripts

data/string_task/             # Train/eval parquet files
eval/                         # Evaluation rollouts and accuracy JSONs
experiments/
  data-onpolicy-loss-fn-vary/ # On-policy × {SFT, GRPO, POS+NEG, REINFORCE+}
  data-teacher-loss-fn-vary/  # Teacher data × {SFT, GRPO, POS+NEG, REINFORCE+}
  data-bootstrap-loss-fn-vary/ # Bootstrap data × {SFT, GRPO, POS+NEG, REINFORCE+}
  util/                       # serve_vllm.sh + generation_client.sh
  eval/                       # Evaluation scripts
deploy/
  launcher.py                 # Submit any experiment script via sbatch template
  template*.sbatch            # Slurm templates (general / A100 / flame / light / preempt / modal)
  serve_vllm.sbatch           # Slurm job for vLLM server
  generation_client.sbatch    # Slurm job for generation client
```

---

## Monitoring

Runs log to W&B under project `string-task`. Key metrics:

| Metric | Description |
|---|---|
| `reward/score/mean` | Mean rollout reward before filtering |
| `training/n_positive_seq` | Positive samples per batch |
| `training/n_negative_seq` | Negative samples per batch (0 for SFT/GRPOMask) |
| `training/weight_mean` | Mean sample weight sent to the actor |
| `training/data_source_is_off_policy` | 1 if using teacher/bootstrap/iterative data |
| `actor/pg_loss` | Loss value |
| `actor/perplexity` | Response perplexity |
| `val/...` | Validation accuracy by level |

Rollout generations are saved to `trainer.rollout_data_dir` and validation rollouts to `trainer.validation_data_dir` every step.
