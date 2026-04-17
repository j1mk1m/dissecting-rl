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

## Unified Loss Framework

All training modes use on-policy rollouts with a binary reward ($r_i \in \{0, 1\}$) and reduce to the same loss form:

$$\ell(\theta) = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G w_i \cdot \log \pi_\theta(y_i \mid x)\right]$$

where the per-sample weight $w_i$ and whether negative samples are used differs by method:

| Loss | Positive weight $w^+$ | Negative weight $w^-$ | `reward_baseline` | Extra flags |
|---|---|---|---|---|
| SFT | $1$ (+ $1/T$ norm) | $0$ | `"none"` | — |
| GRPOMask | $(1-\mu)/\sigma$ | $0$ | `"grpo_mask"` | — |
| POS+NEG | $1$ | $-1$ | `"none"` | `enable_negative_sample_training: true` |
| REINFORCE+ | $1-\mu$ | $-\mu$ | `"mean"` | — |
| GRPO | $(1-\mu)/\sigma$ | $-\mu/\sigma$ | `"mean"` | `reward_normalize_std: true` |

where $\mu$ and $\sigma$ are the mean and standard deviation of rewards within each rollout group.

**Token aggregation** (`actor_rollout_ref.actor.loss_agg_mode`):
- SFT: `seq-mean-token-mean` — divides by sequence length $T$, matching the SFT $1/T$ formula
- All others: `seq-mean-token-sum` — applies the scalar $w_i$ to the full sequence log-prob

---

## Setup

```bash
git clone <this-repo>
cd compositional-generality
pip install -e ".[vllm]"
conda activate osft
```

**Requirements:** 8× A100 80GB GPUs (for the standard scripts). The test script (`osft_test.sh`) runs on 4 GPUs with smaller batch sizes.

---

## Data

| Path | Description |
|---|---|
| `data/string_task/stage2_level2/train.parquet` | Train set, depth-2 compositions |
| `data/string_task/stage2_level1to8/test.parquet` | Eval set, depths 1–8 |
| `data/string_task/stage2_level3/train.parquet` | Train set, depth-3 compositions |
| `data/string_task/level2_balanced/eval.parquet` | Balanced depth-2 eval (14 400 problems) |

To regenerate datasets:
```bash
python scripts/data_preprocess/string_data.py
python scripts/data_preprocess/string_data_analysis.py --input <parquet>
```

---

## Training

All methods use the OSFT recipe (`recipe.osft.main_osft`) with the backbone `gyeongwk/stage1-rft` (Llama-3.1-8B fine-tuned on depth-1 data via RFT).

### Quick start

```bash
# Pick one method and run directly:
bash experiments/osft/sft.sh
bash experiments/osft/grpo_mask.sh
bash experiments/osft/pos_neg.sh
bash experiments/osft/reinforce_plus.sh
bash experiments/osft/grpo.sh
```

### Cluster (Slurm)

```bash
sbatch deploy/osft/sft.sbatch
sbatch deploy/osft/grpo_mask.sbatch
sbatch deploy/osft/pos_neg.sbatch
sbatch deploy/osft/reinforce_plus.sbatch
sbatch deploy/osft/grpo.sbatch
```

### Key hyperparameters

| Parameter | Value |
|---|---|
| Backbone | `gyeongwk/stage1-rft` (Llama-3.1-8B) |
| Rollouts per prompt ($G$) | 16 |
| Train batch size | 16 prompts |
| Max response length | 8192 tokens |
| Learning rate | 1e-6 with 5 warmup steps |
| Epochs | 1 |
| Validation frequency | every 25 steps |

### Resuming from checkpoint

```bash
python3 -m recipe.osft.main_osft \
    ... \
    trainer.resume_mode=auto \
    trainer.default_local_dir=<checkpoint_dir>
```

---

## Repository Structure

```
verl/                         # verl framework (Ray, FSDP, vLLM rollout)
recipe/osft/                  # Training recipe (builds on verl)
  main_osft.py                # Entry point
  osft_trainer.py             # RayOSFTTrainer training loop
  osft_sample_selection.py    # Per-sample weight computation (all 5 losses)
  dp_actor.py                 # Weighted NLL loss + backward
  config/osft_trainer.yaml    # All configurable options

scripts/                      # Helpful utility scripts
  data_preprocess/
    string_data.py            # Dataset generation (25 operators, compositions)
    string_data_analysis.py   # Composition distribution analysis
  analysis/
  generation/
  evaluation/

data/                         # Train/eval dataset parquet files 
eval/                         # Stores evaluation results
experiments/                  # Bash scripts to run experiments
deploy/                       # Slurm sbatch wrappers 
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
| `actor/pg_loss` | Loss value |
| `actor/perplexity` | Response perplexity |
| `val/...` | Validation accuracy by level |

Rollout generations are saved to `trainer.rollout_data_dir` and validation rollouts to `trainer.validation_data_dir` every step.
