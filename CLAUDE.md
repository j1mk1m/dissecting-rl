# CLAUDE.md

## Project Overview

This is a **compositional generality research project** built on top of [verl](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLM) by ByteDance. The core research question is: *can LLMs generalize compositionally on string manipulation tasks?*

The project trains LLMs (e.g., Llama-3.1-8B) using Offline SFT with RL (OSFT) on string transformation tasks of varying complexity (composition depth), then evaluates generalization to unseen compositions.

---

## Repository Structure

```
compositional-generality/
├── verl/                    # Core verl framework (distributed RL training)
│   ├── trainer/             # Main training entrypoints & Hydra configs
│   ├── workers/             # Actor, Critic, Rollout, RewardManager workers
│   ├── models/              # Model implementations
│   ├── utils/               # Dataset, reward scoring, checkpointing, profiling
│   └── experimental/        # Async/off-policy training variants
├── recipe/
│   ├── osft/                # Offline SFT trainer (main recipe used here)
│   └── dpo/                 # DPO trainer
├── scripts/
│   └── data_preprocess/     # Dataset generation and analysis scripts
│       ├── string_data.py           # Generates string task datasets
│       ├── string_data_analysis.py  # Analyzes composition distributions
│       ├── string_manipulation_sft.py
│       └── split_data_into_rft_data.py
├── experiments/
│   ├── osft/                # Shell scripts for OSFT training runs
│   │   └── osft_string_task.sh  # Primary experiment script
│   ├── grpo/                # GRPO experiment scripts
│   └── dpo/                 # DPO experiment scripts
└── data/
    ├── string_task/         # String manipulation task datasets
    │   ├── stage2_level2/       # Level 2 train (2-op compositions)
    │   ├── stage2_level3/       # Level 3 train (3-op compositions)
    │   ├── level2_balanced/     # Balanced level 2 (14400 samples)
    │   ├── level2_with_code/    # Level 2 with code context
    │   ├── teacher-rl-checkpoint/   # Rollout data from RL teacher
    │   └── teacher-base-with-code/  # Rollout data from base teacher
    └── benchmarks/          # Math benchmarks (AIME, AMC, Math500, etc.)
```

---

## String Task Domain

The string manipulation task trains models to predict outputs of Python functions that compose 25 primitive string operators:

| ID | Function | Description |
|----|----------|-------------|
| func_0 | deterministic_shuffle | Reorder chars via fixed multiplier permutation |
| func_1 | repeat_str | Repeat string n times |
| func_2 | remove_vowels | Remove vowels |
| func_3 | sort_chars | Sort characters |
| func_4 | reverse_words | Reverse word order |
| func_5 | add_prefix | Add fixed prefix |
| func_6 | add_suffix | Add fixed suffix |
| func_7 | interlace_str | Interlace two strings char-by-char |
| func_8 | rotate_str | Rotate string by n positions |
| func_9 | mirror_str | Mirror/palindrome operation |
| func_10 | alternate_case | Alternate upper/lowercase |
| func_11 | shift_chars | Shift characters by offset |
| func_12 | vowel_to_number | Replace vowels with digits |
| func_13 | insert_separator | Insert separator between chars |
| func_14 | duplicate_every_char | Duplicate each character |
| func_15 | fancy_brackets | Wrap with brackets |
| func_16 | compress_repeats | Compress repeated chars |
| func_17 | recursive_reverse | Recursive string reversal |
| func_18 | loop_concat | Loop-based concatenation |
| func_19 | while_rotate | While-loop rotation |
| func_20 | recursive_interlace | Recursive interlacing |
| func_21 | loop_filter_nonalpha | Filter non-alpha via loop |
| func_22 | verify_even_length | Even-length verification |
| func_23 | backchain_add_digit | Backchain digit addition |
| func_24 | backchain_palindrome | Backchain palindrome check |

**Prompt format** (`FORWARD_PROMPT` in `scripts/data_preprocess/string_data.py`):
```
You are given a code:

{code}

Can you predict the output of `main_solution("{input}")` without writing any code?
Please reason and put your final answer in the following json format: {"output": <your output>},
where <your output> should be the final string.
```

Levels correspond to composition depth (number of operators applied sequentially).

---

## Running Experiments

### Primary OSFT Training (string task)

```bash
cd /home/gyeongwk/compositional-generality
bash experiments/osft/osft_string_task.sh
```

Key parameters in the script:
- `BACKBONE_PATH=gyeongwk/stage1-rft` — HuggingFace model path
- `TRAIN_FILE` — points to `stage2_level2/train.parquet`
- `VAL_FILE` — points to `stage2_level1to8/test.parquet`
- `ROLLOUT_N=16` — samples per prompt during training
- `LR=1e-6`, `MAX_GEN_LENGTH=8192`
- Output goes to `/data/user_data/gyeongwk/checkpoints/string-task/`

### Direct Python invocation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m recipe.osft.main_osft \
    data.train_files=<path>.parquet \
    data.val_files=<path>.parquet \
    actor_rollout_ref.model.path=<hf_model_or_path> \
    trainer.project_name=<wandb_project> \
    trainer.experiment_name=<run_name> \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=1
```

Configuration uses Hydra — override any YAML key via CLI dot notation.

### Data Generation

```bash
python scripts/data_preprocess/string_data.py --help
python scripts/data_preprocess/string_data_analysis.py --input <parquet_file>
```

---

## Framework Details (verl)

### Key Concepts

- **DataProto**: Unified data structure wrapping TensorDict for passing data between workers
- **Ray**: All distributed workers are Ray actors; single controller (driver) orchestrates them
- **Hydra**: All configs live in `verl/trainer/config/` and `recipe/osft/config/`; override via CLI
- **FSDP2**: Default parallelization strategy for training

### Configuration System

Main OSFT config: `recipe/osft/config/osft_trainer.yaml`
PPO config (if using verl PPO): `verl/trainer/config/ppo_trainer.yaml`

Config hierarchy (Hydra defaults list):
- `actor/dp_actor` → FSDP-based actor
- `rollout/rollout` → rollout config
- `engine/fsdp` → FSDP engine
- `optim/fsdp` → optimizer

### Worker Architecture

```
Driver (main_osft.py)
├── ActorRolloutRefWorker  (policy + rollout + reference model)
└── RewardManager          (reward computation)
```

### Reward Manager

The reward function for string tasks is exact-match on the predicted output string. Reward managers live in `verl/workers/reward_manager/`.

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | - | Core DL framework |
| Ray | >=2.41.0 | Distributed worker orchestration |
| vLLM | >=0.8.5, <=0.12.0 | Inference engine for rollout |
| Hydra | - | Config management |
| TensorDict | >=0.8.0, <=0.10.0, !=0.9.0 | Tensor data structures |
| transformers | - | HuggingFace model loading |
| wandb | - | Experiment tracking |

Install:
```bash
pip install -e ".[vllm]"          # with vLLM
pip install -e ".[vllm,math]"     # with math reward scoring
```

---

## Important Environment Variables

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `HYDRA_FULL_ERROR=1` | Show full Hydra config errors |
| `TOKENIZERS_PARALLELISM` | Tokenizer threading |
| `NCCL_DEBUG` | NCCL communication debugging |
| `VLLM_LOGGING_LEVEL` | vLLM log verbosity |

---

## Data Format

Training parquet files have columns:
- `prompt`: list of chat messages (HuggingFace format)
- `reward_model`: dict with reward metadata
- `extra_info`: dict with ground truth and task metadata

See `verl/utils/dataset/README.md` for full schema.

---

## Checkpoints

Checkpoints are saved to `trainer.default_local_dir` (set in the shell script to `/data/user_data/gyeongwk/checkpoints/`). The `trainer.save_freq` controls how often. Rollout data is saved separately to `trainer.rollout_data_dir`.

To resume training: set `trainer.resume_mode=auto` and point `trainer.default_local_dir` to the existing checkpoint directory.
