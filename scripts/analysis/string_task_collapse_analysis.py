"""Investigate model collapse in string-task off-policy (Bootstrap/Teacher) runs
that use a negative-sample loss (POS+NEG, REINFORCE+BASELINE, GRPO).

Runs analyzed live in /data/user_data/gyeongwk/checkpoints/string-task/:
  - Bootstrap-GRPO
  - Bootstrap-REINFORCE+BASELINE
  - Teacher-GRPO
  - Teacher-REINFORCE+BASELINE
(Bootstrap-POS+NEG / Teacher-POS+NEG were never checkpointed - not analyzed.)

Both "Bootstrap" and "Teacher" experiments use trainer.data_source.mode=Teacher
(recipe/osft/data_source_controller.py): a frozen, pre-generated pool of 16
rollouts/prompt, cycled repeatedly with no live resampling. The only
difference between the two families is which frozen pool:
  - Bootstrap: data/string_task/teacher-bootstrap/rollout.parquet
      (rollouts from the *untrained base* stage1-rft model)
  - Teacher:   data/string_task/teacher-grpo/rollout.parquet
      (rollouts distilled from an RL-trained teacher checkpoint)

This script:
  1. Loads rollout_eval_data/*.jsonl (temp=0 validation rollouts, logged every
     ~100 steps) for each run -> avg response length & accuracy trajectory.
  2. Pulls concrete example generations at early/mid/late steps to see the
     collapse in the model's own words.
  3. Scores the two frozen pools directly (same extraction/matching logic as
     verl/utils/reward_score/codeio.py's compute_score_forward) and tokenizes
     every response with the actual training tokenizer (gyeongwk/stage1-rft)
     to get real token lengths, then reports the pos/neg length gap and
     group-composition (all-correct / all-incorrect / mixed) stats that
     determine how much gradient signal each loss actually gets and in which
     direction it points.

Usage:
    python scripts/analysis/string_task_collapse_analysis.py \
        --checkpoints-dir /data/user_data/gyeongwk/checkpoints/string-task \
        --output-dir reports/figures
"""

import argparse
import csv
import glob
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = {
    "Bootstrap-GRPO": "Bootstrap-GRPO",
    "Bootstrap-REINFORCE+BASELINE": "Bootstrap-REINFORCE+BASELINE",
    "Teacher-GRPO": "Teacher-GRPO",
    "Teacher-REINFORCE+BASELINE": "Teacher-REINFORCE+BASELINE",
}

POOLS = {
    "Bootstrap": "data/string_task/teacher-bootstrap/rollout.parquet",
    "Teacher": "data/string_task/teacher-grpo/rollout.parquet",
}

TOKENIZER_PATH = "gyeongwk/stage1-rft"
MAX_GEN_LENGTH = 4096


# ---------------------------------------------------------------------------
# 1. rollout_eval_data trajectories
# ---------------------------------------------------------------------------

def load_eval_stats(run_dir: str):
    files = glob.glob(os.path.join(run_dir, "rollout_eval_data", "*.jsonl"))
    out = []
    for fp in files:
        step = int(os.path.basename(fp).replace(".jsonl", ""))
        lens, scores = [], []
        with open(fp) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                lens.append(d["response_lengths"])
                scores.append(d["score"])
        if not lens:
            continue
        out.append((step, sum(lens) / len(lens), sum(scores) / len(scores), len(lens)))
    out.sort()
    return out


def save_step_csv(step_stats, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "avg_len", "avg_score", "n"])
        for row in step_stats:
            w.writerow(row)


def plot_eval_trajectories(all_stats: dict, output_path: str):
    fig, (ax_len, ax_acc) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for label, rows in all_stats.items():
        xs = [r[0] for r in rows]
        lens = [r[1] for r in rows]
        accs = [r[2] for r in rows]
        ax_len.plot(xs, lens, marker="o", linewidth=1.8, label=label)
        ax_acc.plot(xs, accs, marker="o", linewidth=1.8, label=label)
    for ax in (ax_len, ax_acc):
        ax.grid(True, alpha=0.3)
    ax_len.set_ylabel("Avg response length (tokens)")
    ax_len.set_title("String-task off-policy runs: rollout_eval_data (temp=0) trajectory")
    ax_len.legend()
    ax_acc.set_ylabel("Avg accuracy (exact-match score)")
    ax_acc.set_xlabel("Training step")
    ax_acc.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Example generations
# ---------------------------------------------------------------------------

def dump_examples(run_dir: str, steps: list, n_examples: int, out_path: str):
    with open(out_path, "w") as out:
        for step in steps:
            fp = os.path.join(run_dir, "rollout_eval_data", f"{step}.jsonl")
            if not os.path.isfile(fp):
                out.write(f"\n=== step {step}: file not found ===\n")
                continue
            with open(fp) as f:
                lines = f.readlines()
            random.Random(0).shuffle(lines)
            out.write(f"\n{'=' * 100}\n=== step {step} (n={len(lines)}) ===\n{'=' * 100}\n")
            for line in lines[:n_examples]:
                d = json.loads(line)
                out.write(f"\n--- score={d['score']} len={d['response_lengths']} ---\n")
                out.write("[PROMPT TAIL]\n" + d["input"][-300:] + "\n")
                out.write("[OUTPUT]\n" + d["output"] + "\n")
    print(f"Saved examples to {out_path}")


# ---------------------------------------------------------------------------
# 3. Frozen-pool scoring: pos/neg length gap + group composition
# ---------------------------------------------------------------------------

def score_pool(parquet_path: str, tokenizer, sample_prompts: int, seed: int = 0):
    import importlib.util
    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "codeio_reward", "verl/utils/reward_score/codeio.py"
    )
    codeio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codeio)
    codeio.do_print = False

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    if sample_prompts and sample_prompts < len(df):
        df = df.sample(n=sample_prompts, random_state=seed)

    all_texts = []
    all_scores = []
    group_sizes = []

    for _, row in df.iterrows():
        gt = json.loads(row["reward_model"]["ground_truth"])
        responses = list(row["responses"])
        scores = []
        for resp in responses:
            try:
                extracted = codeio.extract_last_complete_json(resp)
                if extracted is None or not isinstance(extracted, dict) or "output" not in extracted:
                    scores.append(0)
                    continue
                scores.append(codeio.compute_score_forward(extracted["output"], gt))
            except Exception:
                scores.append(0)
        all_texts.extend(responses)
        all_scores.extend(scores)
        group_sizes.append(len(responses))

    # Tokenize in one batch for speed.
    enc = tokenizer(all_texts, add_special_tokens=False)
    lengths = [len(ids) for ids in enc["input_ids"]]

    return all_texts, all_scores, lengths, group_sizes


def summarize_pool(label: str, scores, lengths, group_sizes, max_gen_length: int):
    print(f"\n=== Frozen pool: {label} ({len(scores)} responses, {len(group_sizes)} prompts) ===")
    pos = [l for l, s in zip(lengths, scores) if s > 0]
    neg = [l for l, s in zip(lengths, scores) if s <= 0]
    print(f"  overall accuracy: {sum(scores) / len(scores):.3f}")
    print(f"  avg correct (pos) length:   {sum(pos) / len(pos):.0f}  (n={len(pos)})" if pos else "  no positives")
    print(f"  avg incorrect (neg) length: {sum(neg) / len(neg):.0f}  (n={len(neg)})" if neg else "  no negatives")
    if pos and neg:
        print(f"  gap (pos - neg): {sum(pos)/len(pos) - sum(neg)/len(neg):+.0f}")
    cap = max_gen_length - 8
    pos_capped = sum(1 for l in pos if l >= cap)
    neg_capped = sum(1 for l in neg if l >= cap)
    print(f"  pos capped (>={cap} tok): {pos_capped}/{len(pos)} ({100*pos_capped/max(1,len(pos)):.1f}%)")
    print(f"  neg capped (>={cap} tok): {neg_capped}/{len(neg)} ({100*neg_capped/max(1,len(neg)):.1f}%)")

    # group composition
    idx = 0
    all_correct = all_incorrect = mixed = 0
    for gsize in group_sizes:
        g_scores = scores[idx:idx + gsize]
        idx += gsize
        c = sum(1 for s in g_scores if s > 0)
        if c == 0:
            all_incorrect += 1
        elif c == gsize:
            all_correct += 1
        else:
            mixed += 1
    n_groups = len(group_sizes)
    print(f"  group composition: all-correct={all_correct} ({100*all_correct/n_groups:.1f}%)  "
          f"all-incorrect={all_incorrect} ({100*all_incorrect/n_groups:.1f}%)  "
          f"mixed={mixed} ({100*mixed/n_groups:.1f}%)")
    return {
        "accuracy": sum(scores) / len(scores),
        "pos_len": sum(pos) / len(pos) if pos else None,
        "neg_len": sum(neg) / len(neg) if neg else None,
        "all_correct_pct": 100 * all_correct / n_groups,
        "all_incorrect_pct": 100 * all_incorrect / n_groups,
        "mixed_pct": 100 * mixed / n_groups,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints-dir", default="/data/user_data/gyeongwk/checkpoints/string-task")
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--pool-sample", type=int, default=3000,
                         help="Number of prompts to sample per frozen pool (0 = all 19200)")
    parser.add_argument("--skip-pools", action="store_true")
    parser.add_argument("--n-examples", type=int, default=6)
    args = parser.parse_args()

    ckpt_dir = args.checkpoints_dir
    out_dir = args.output_dir

    # --- 1. eval trajectories ---
    all_stats = {}
    for label, subdir in RUNS.items():
        run_dir = os.path.join(ckpt_dir, subdir)
        if not os.path.isdir(run_dir):
            print(f"skip {label}: {run_dir} not found")
            continue
        stats = load_eval_stats(run_dir)
        all_stats[label] = stats
        save_step_csv(stats, os.path.join(out_dir, f"string_step_stats_{label.replace('+', '').replace(' ', '_')}.csv"))
        print(f"\n### {label} (rollout_eval_data trajectory) ###")
        for step, avg_len, avg_score, n in stats:
            print(f"  step {step:>5}: avg_len={avg_len:7.1f}  avg_score={avg_score:.3f}  n={n}")

    plot_eval_trajectories(all_stats, os.path.join(out_dir, "string_task_offpolicy_collapse.png"))

    # --- 2. example generations ---
    examples_dir = os.path.join(out_dir, "string_task_examples")
    os.makedirs(examples_dir, exist_ok=True)
    for label, subdir in RUNS.items():
        run_dir = os.path.join(ckpt_dir, subdir)
        if not os.path.isdir(run_dir):
            continue
        steps = sorted(s for s, _, _, _ in all_stats.get(label, []))
        if not steps:
            continue
        pick = sorted(set([steps[0], steps[len(steps) // 2], steps[-1]]))
        dump_examples(run_dir, pick, args.n_examples,
                      os.path.join(examples_dir, f"{label.replace('+', '')}.txt"))

    # --- 3. frozen pool analysis ---
    if not args.skip_pools:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
        pool_summary = {}
        for label, path in POOLS.items():
            if not os.path.isfile(path):
                print(f"skip pool {label}: {path} not found")
                continue
            texts, scores, lengths, group_sizes = score_pool(path, tokenizer, args.pool_sample)
            pool_summary[label] = summarize_pool(label, scores, lengths, group_sizes, MAX_GEN_LENGTH)
        with open(os.path.join(out_dir, "string_pool_summary.json"), "w") as f:
            json.dump(pool_summary, f, indent=2)


if __name__ == "__main__":
    main()
