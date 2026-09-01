"""Reproduce all findings in reports/math_task_response_length_dynamics.md.

Recomputes, directly from raw rollout_data/rollout_eval_data JSONL files under a
math-task checkpoints directory:
  1. Per-run, per-step average response length and accuracy (on-policy runs).
  2. Per-step correct-vs-incorrect ("positive vs negative sample") average length
     for the on-policy GRPO run.
  3. Per-prompt-group (GRPO advantage unit, 16 same-prompt rollouts) composition
     analysis: outcome mix (all-correct / all-incorrect / mixed), success-rate
     buckets within mixed groups, and the GRPO std-normalized advantage-weighted
     average correct-sample length.
  4. Bootstrap-GRPO's rollout_eval_data trajectory (length/accuracy collapse).

Also renders the length+accuracy figures referenced in the report.

Usage:
    python scripts/analysis/math_task_length_dynamics_analysis.py
    python scripts/analysis/math_task_length_dynamics_analysis.py \
        --checkpoints-dir /data/user_data/gyeongwk/checkpoints/math-task \
        --output-dir reports/figures

Each step-level aggregate is also cached as a CSV under --output-dir so the
raw per-step numbers can be re-plotted or re-analyzed without re-reading the
(large, slow to parse) rollout_data directories.
"""

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = {
    "GRPO": "math-onpolicy-GRPO-Qwen3-1.7B",
    "PosNeg": "math-onpolicy-PosNeg-Qwen3-1.7B",
    "Reinforce-Baseline": "math-onpolicy-Reinforce-Baseline-Qwen3-1.7B",
    "SFT": "math-onpolicy-SFT-Qwen3-1.7B",
}
BOOTSTRAP_RUN = "math-bootstrap-GRPO-easy-Qwen3-1.7B"

EASY_MEDIUM_BOUNDARY_STEP = 660  # 10,560 easy rows / batch_size 16
WINDOW = 50


# ---------------------------------------------------------------------------
# 1. Per-step length/accuracy aggregation (on-policy runs)
# ---------------------------------------------------------------------------

def load_rollout_step_stats(run_dir: str):
    """Return sorted list of (step, avg_len, avg_score, n, pos_len, neg_len) from rollout_data/*.jsonl.

    pos_len/neg_len are the mean response_lengths of correct (score>=0.5) and
    incorrect (score<0.5) samples in that step; None if a class is empty.
    """
    files = glob.glob(os.path.join(run_dir, "rollout_data", "*.jsonl"))
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
                l, s = d.get("response_lengths"), d.get("score")
                if l is None or s is None:
                    continue
                lens.append(l)
                scores.append(s)
        if not lens:
            continue
        pos = [l for l, s in zip(lens, scores) if s >= 0.5]
        neg = [l for l, s in zip(lens, scores) if s < 0.5]
        out.append((
            step,
            sum(lens) / len(lens),
            sum(scores) / len(scores),
            len(lens),
            (sum(pos) / len(pos)) if pos else None,
            (sum(neg) / len(neg)) if neg else None,
        ))
    out.sort()
    return out


def load_bootstrap_eval_stats(run_dir: str):
    """Return sorted list of (step, avg_len, avg_score) from rollout_eval_data/*.jsonl."""
    files = glob.glob(os.path.join(run_dir, "rollout_eval_data", "*.jsonl"))
    out = []
    for fp in files:
        step = int(os.path.basename(fp).replace(".jsonl", ""))
        lens, scores = [], []
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                lens.append(d["response_lengths"])
                scores.append(d["score"])
        out.append((step, sum(lens) / len(lens), sum(scores) / len(scores)))
    out.sort()
    return out


def windowed(step_stats, window=WINDOW):
    """Collapse a (step, avg_len, avg_score, n, ...) list into `window`-step bins."""
    rows = []
    for i in range(0, len(step_stats), window):
        chunk = step_stats[i:i + window]
        steps = [c[0] for c in chunk]
        avg_len = sum(c[1] for c in chunk) / len(chunk)
        avg_score = sum(c[2] for c in chunk) / len(chunk)
        rows.append((steps[0], steps[-1], avg_len, avg_score))
    return rows


def save_step_csv(step_stats, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "avg_len", "avg_score", "n", "pos_len", "neg_len"])
        for row in step_stats:
            w.writerow(row)


# ---------------------------------------------------------------------------
# 2. Per-prompt-group (GRPO advantage unit) composition analysis
# ---------------------------------------------------------------------------

def load_groups_for_steps(run_dir: str, steps_range: set, group_size: int = 16):
    """Yield (n_correct, lens, scores) for every prompt-group (consecutive
    `group_size`-row block sharing the same prompt) in the given steps."""
    files = glob.glob(os.path.join(run_dir, "rollout_data", "*.jsonl"))
    files = [f for f in files if int(Path(f).stem) in steps_range]
    for fp in files:
        with open(fp) as f:
            lines = [json.loads(l) for l in f]
        for i in range(0, len(lines), group_size):
            g = lines[i:i + group_size]
            if len(g) < group_size:
                continue
            lens = [x["response_lengths"] for x in g]
            scores = [x["score"] for x in g]
            n_correct = sum(1 for s in scores if s >= 0.5)
            yield n_correct, lens, scores


def group_composition_summary(run_dir: str, steps_range: set, label: str):
    """Print the outcome-composition, success-rate-bucket, and advantage-weighted
    correct-length tables for one phase (a set of steps)."""
    groups = list(load_groups_for_steps(run_dir, steps_range))
    n_total = len(groups)
    all_correct = sum(1 for g in groups if g[0] == 16)
    all_incorrect = sum(1 for g in groups if g[0] == 0)
    mixed = [g for g in groups if 0 < g[0] < 16]

    print(f"\n=== {label} ===")
    print(f"  total groups={n_total}  all-correct={all_correct}  "
          f"all-incorrect={all_incorrect}  mixed={len(mixed)}")

    if not mixed:
        return

    low = [g for g in mixed if g[0] <= 4]
    mid = [g for g in mixed if 5 <= g[0] <= 11]
    high = [g for g in mixed if g[0] >= 12]
    n = len(mixed)
    print(f"  mixed-group success-rate composition: "
          f"low(1-4/16)={100*len(low)/n:.1f}%  mid(5-11/16)={100*len(mid)/n:.1f}%  "
          f"high(12-15/16)={100*len(high)/n:.1f}%")

    for bucket_name, bucket in [("low(1-4/16)", low), ("mid(5-11/16)", mid), ("high(12-15/16)", high)]:
        if not bucket:
            continue
        pos_lens = [l for n_c, lens, scores in bucket for l, s in zip(lens, scores) if s >= 0.5]
        neg_lens = [l for n_c, lens, scores in bucket for l, s in zip(lens, scores) if s < 0.5]
        print(f"    {bucket_name}: n_groups={len(bucket)}  "
              f"avg_correct_len={sum(pos_lens)/len(pos_lens):.0f}  "
              f"avg_incorrect_len={sum(neg_lens)/len(neg_lens):.0f}")

    # Plain vs GRPO-advantage-weighted average correct length.
    plain_sum = plain_n = 0
    adv_num = adv_den = 0.0
    for n_correct, lens, scores in mixed:
        p = n_correct / 16
        std = math.sqrt(p * (1 - p))
        adv_pos = (1 - p) / std
        for l, s in zip(lens, scores):
            if s >= 0.5:
                plain_sum += l
                plain_n += 1
                adv_num += adv_pos * l
                adv_den += adv_pos
    print(f"  plain avg correct length (unweighted):     {plain_sum/plain_n:.0f}")
    print(f"  GRPO-advantage-weighted avg correct length: {adv_num/adv_den:.0f}")

    # Fraction of samples truncated at the generation cap (>=4090 tokens).
    pos_total = neg_total = pos_capped = neg_capped = 0
    for n_correct, lens, scores in mixed:
        for l, s in zip(lens, scores):
            capped = l >= 4090
            if s >= 0.5:
                pos_total += 1
                pos_capped += capped
            else:
                neg_total += 1
                neg_capped += capped
    print(f"  within mixed groups -> pos capped {pos_capped}/{pos_total} "
          f"({100*pos_capped/max(1,pos_total):.1f}%), "
          f"neg capped {neg_capped}/{neg_total} ({100*neg_capped/max(1,neg_total):.1f}%)")


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------

def plot_onpolicy_runs(all_windowed: dict, output_path: str):
    fig, (ax_len, ax_acc) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for label, rows in all_windowed.items():
        xs = [(a + b) / 2 for a, b, _, _ in rows]
        lens = [l for _, _, l, _ in rows]
        accs = [s for _, _, _, s in rows]
        ax_len.plot(xs, lens, marker="", linewidth=1.8, label=label)
        ax_acc.plot(xs, accs, marker="", linewidth=1.8, label=label)

    for ax in (ax_len, ax_acc):
        ax.axvline(EASY_MEDIUM_BOUNDARY_STEP, color="gray", linestyle="--", linewidth=1,
                   label="_nolegend_")
        ax.grid(True, alpha=0.3)

    ax_len.text(EASY_MEDIUM_BOUNDARY_STEP + 10, ax_len.get_ylim()[1] * 0.95,
                "easy→medium", fontsize=9, color="gray")
    ax_len.set_ylabel("Avg response length (tokens)")
    ax_len.set_title("On-policy math-task runs: response length & accuracy vs. training step\n"
                      "(50-step windows, dashed line = easy→medium data transition)")
    ax_len.legend()

    ax_acc.set_ylabel("Avg accuracy (score)")
    ax_acc.set_xlabel("Training step")
    ax_acc.legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_bootstrap_run(bootstrap_stats, output_path: str):
    steps = [s for s, _, _ in bootstrap_stats]
    lens = [l for _, l, _ in bootstrap_stats]
    accs = [a for _, _, a in bootstrap_stats]

    fig, (ax_len, ax_acc) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_len.plot(steps, lens, marker="o", linewidth=1.8, color="firebrick")
    ax_len.set_ylabel("Avg response length (tokens)")
    ax_len.set_title("Bootstrap-GRPO (teacher mode): response length & accuracy collapse\n"
                      "(rollout_eval_data, frozen off-policy training data)")
    ax_len.grid(True, alpha=0.3)

    ax_acc.plot(steps, accs, marker="o", linewidth=1.8, color="firebrick")
    ax_acc.set_ylabel("Avg accuracy (score)")
    ax_acc.set_xlabel("Training step")
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints-dir", default="/data/user_data/gyeongwk/checkpoints/math-task",
                         help="Directory containing the per-run checkpoint subfolders")
    parser.add_argument("--output-dir", default="reports/figures",
                         help="Where to save figures and per-step CSV caches")
    args = parser.parse_args()

    ckpt_dir = args.checkpoints_dir
    out_dir = args.output_dir

    # --- 1+2. Per-step stats for on-policy runs ---
    all_step_stats = {}
    for label, subdir in RUNS.items():
        run_dir = os.path.join(ckpt_dir, subdir)
        if not os.path.isdir(run_dir):
            print(f"skip {label}: {run_dir} not found")
            continue
        stats = load_rollout_step_stats(run_dir)
        all_step_stats[label] = stats
        save_step_csv(stats, os.path.join(out_dir, f"step_stats_{label.replace(' ', '_')}.csv"))

        print(f"\n### {label} (50-step windows) ###")
        for a, b, avg_len, avg_score in windowed(stats):
            print(f"  steps {a:>4}-{b:>4}: avg_len={avg_len:6.0f}  avg_score={avg_score:.3f}")

    # --- GRPO positive-vs-negative sample length gap ---
    if "GRPO" in all_step_stats:
        print("\n### GRPO: pos_len vs neg_len per 50-step window ###")
        grpo_stats = all_step_stats["GRPO"]
        for i in range(0, len(grpo_stats), WINDOW):
            chunk = grpo_stats[i:i + WINDOW]
            steps = [c[0] for c in chunk]
            pos_vals = [(c[4], c[3]) for c in chunk if c[4] is not None]
            neg_vals = [(c[5], c[3]) for c in chunk if c[5] is not None]
            if not pos_vals or not neg_vals:
                continue
            pos_avg = sum(v * n for v, n in pos_vals) / sum(n for _, n in pos_vals)
            neg_avg = sum(v * n for v, n in neg_vals) / sum(n for _, n in neg_vals)
            print(f"  steps {steps[0]:>4}-{steps[-1]:>4}: pos_len={pos_avg:6.0f}  "
                  f"neg_len={neg_avg:6.0f}  gap={pos_avg-neg_avg:+6.0f}")

    # --- 3. Per-prompt-group composition analysis (GRPO) ---
    grpo_dir = os.path.join(ckpt_dir, RUNS["GRPO"])
    if os.path.isdir(grpo_dir):
        group_composition_summary(grpo_dir, set(range(100, 301)), "GRPO: EASY PHASE (steps 100-300)")
        group_composition_summary(grpo_dir, set(range(700, 901)), "GRPO: FIRST CRASH (steps 700-900)")

    # --- 4. Bootstrap-GRPO eval trajectory ---
    bootstrap_dir = os.path.join(ckpt_dir, BOOTSTRAP_RUN)
    bootstrap_stats = None
    if os.path.isdir(bootstrap_dir):
        bootstrap_stats = load_bootstrap_eval_stats(bootstrap_dir)
        print("\n### Bootstrap-GRPO rollout_eval_data trajectory ###")
        for step, avg_len, avg_score in bootstrap_stats:
            print(f"  step {step:>4}: avg_len={avg_len:7.1f}  avg_score={avg_score:.3f}")
        with open(os.path.join(out_dir, "step_stats_Bootstrap_GRPO_eval.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "avg_len", "avg_score"])
            for row in bootstrap_stats:
                w.writerow(row)

    # --- Plots ---
    all_windowed = {label: windowed(stats) for label, stats in all_step_stats.items()}
    plot_onpolicy_runs(all_windowed, os.path.join(out_dir, "onpolicy_length_accuracy.png"))
    if bootstrap_stats:
        plot_bootstrap_run(bootstrap_stats, os.path.join(out_dir, "bootstrap_grpo_collapse.png"))


if __name__ == "__main__":
    main()
