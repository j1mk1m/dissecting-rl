#!/usr/bin/env python3
"""
Quantitative analysis of rollout data across training steps.

Produces:
  - Accuracy over steps (overall + by depth bucket)
  - Response length over steps
  - Per-function accuracy at final step
  - Format error rate over steps
  - Accuracy vs. response length correlation
  - Cross-run comparison (when --all-runs is set)

Usage:
    python scripts/analysis/analyze_rollouts.py --run Bootstrap-GRPO
    python scripts/analysis/analyze_rollouts.py --run Bootstrap-GRPO --eval
    python scripts/analysis/analyze_rollouts.py --all-runs --eval
    python scripts/analysis/analyze_rollouts.py --run Bootstrap-GRPO --eval --out plots/
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR = "/data/user_data/gyeongwk/checkpoints"

FUNC_NAMES = {
    0: "deterministic_shuffle", 1: "repeat_str", 2: "remove_vowels",
    3: "sort_chars", 4: "reverse_words", 5: "add_prefix", 6: "add_suffix",
    7: "interlace_str", 8: "rotate_str", 9: "mirror_str",
    10: "alternate_case", 11: "shift_chars", 12: "vowel_to_number",
    13: "insert_separator", 14: "duplicate_every_char", 15: "fancy_brackets",
    16: "compress_repeats", 17: "recursive_reverse", 18: "loop_concat",
    19: "while_rotate", 20: "recursive_interlace", 21: "loop_filter_nonalpha",
    22: "verify_even_length", 23: "backchain_add_digit", 24: "backchain_palindrome",
}

DEPTH_BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 100)]


def depth_bucket_label(d: int) -> str:
    for lo, hi in DEPTH_BUCKETS:
        if lo <= d <= hi:
            return f"{lo}" if lo == hi else f"{lo}-{hi}"
    return f"{d}"


def get_composition_depth(input_str: str) -> int:
    m = re.search(r"return (.+)", input_str)
    if not m:
        return 0
    return len(re.findall(r"func_\d+", m.group(1)))


def get_func_ids(input_str: str) -> list[int]:
    m = re.search(r"return (.+)", input_str)
    if not m:
        return []
    return [int(x) for x in re.findall(r"func_(\d+)", m.group(1))]


def has_valid_format(output: str) -> bool:
    return bool(re.search(r'\{"output":', output))


def load_run_data(run_dir: str, is_eval: bool) -> pd.DataFrame:
    subdir = "rollout_eval_data" if is_eval else "rollout_data"
    data_dir = os.path.join(run_dir, subdir)
    if not os.path.isdir(data_dir):
        return pd.DataFrame()

    rows = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl") or fname == "tmp.jsonl":
            continue
        step = int(fname.replace(".jsonl", ""))
        with open(os.path.join(data_dir, fname)) as f:
            for line in f:
                e = json.loads(line)
                depth = get_composition_depth(e["input"])
                funcs = get_func_ids(e["input"])
                rows.append({
                    "step": step,
                    "score": float(e["score"]),
                    "response_lengths": int(e["response_lengths"]),
                    "depth": depth,
                    "bucket": depth_bucket_label(depth),
                    "funcs": funcs,
                    "has_format": has_valid_format(e["output"]),
                })

    return pd.DataFrame(rows)


def save_fig(fig, out_dir: str, name: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Individual run plots ────────────────────────────────────────────────────

def plot_accuracy_over_steps(df: pd.DataFrame, run_name: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = sorted(df["step"].unique())
    overall = [df[df.step == s]["score"].mean() for s in steps]
    ax.plot(steps, overall, "k-o", label="Overall", linewidth=2, markersize=5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{run_name} — Accuracy Over Steps")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_dir, "accuracy_over_steps.png")


def plot_accuracy_by_depth(df: pd.DataFrame, run_name: str, out_dir: str):
    steps = sorted(df["step"].unique())
    bucket_order = []
    for lo, hi in DEPTH_BUCKETS:
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        if label in df["bucket"].values:
            bucket_order.append(label)

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(bucket_order))
    for i, bucket in enumerate(bucket_order):
        sub = df[df["bucket"] == bucket]
        accs = [sub[sub.step == s]["score"].mean() for s in steps]
        n = len(sub) // len(steps) if steps else 0
        ax.plot(steps, accs, "-o", label=f"depth {bucket} (n≈{n})", color=cmap(i),
                markersize=4, linewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{run_name} — Accuracy by Composition Depth")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    save_fig(fig, out_dir, "accuracy_by_depth.png")


def plot_response_length(df: pd.DataFrame, run_name: str, out_dir: str):
    steps = sorted(df["step"].unique())
    correct = [df[(df.step == s) & (df.score == 1)]["response_lengths"].mean() for s in steps]
    wrong = [df[(df.step == s) & (df.score == 0)]["response_lengths"].mean() for s in steps]
    overall = [df[df.step == s]["response_lengths"].mean() for s in steps]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, overall, "k-o", label="Overall", linewidth=2, markersize=5)
    ax.plot(steps, correct, "g-s", label="Correct", linewidth=1.5, markersize=4)
    ax.plot(steps, wrong, "r-^", label="Wrong", linewidth=1.5, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Response Length (tokens)")
    ax.set_title(f"{run_name} — Response Length Over Steps")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_dir, "response_length.png")


def plot_format_error_rate(df: pd.DataFrame, run_name: str, out_dir: str):
    steps = sorted(df["step"].unique())
    fmt_rates = [df[df.step == s]["has_format"].mean() for s in steps]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, fmt_rates, "b-o", linewidth=2, markersize=5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Fraction with valid {\"output\": ...} format")
    ax.set_title(f"{run_name} — Format Compliance Over Steps")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    save_fig(fig, out_dir, "format_compliance.png")


def plot_per_func_accuracy(df: pd.DataFrame, run_name: str, out_dir: str):
    # Use the last available step
    last_step = df["step"].max()
    sub = df[df.step == last_step].copy()

    func_acc = defaultdict(list)
    func_n = defaultdict(int)
    for _, row in sub.iterrows():
        for fid in row["funcs"]:
            func_acc[fid].append(row["score"])
            func_n[fid] += 1

    if not func_acc:
        return

    func_ids = sorted(func_acc.keys())
    accs = [np.mean(func_acc[f]) for f in func_ids]
    ns = [func_n[f] for f in func_ids]
    labels = [f"f{fid}\n{FUNC_NAMES.get(fid,'?')[:12]}" for fid in func_ids]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(func_ids)), accs, color="steelblue", alpha=0.8)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={n}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(func_ids)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Accuracy (presence in composition)")
    ax.set_title(f"{run_name} — Per-Function Accuracy (step {last_step})")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, out_dir, "per_func_accuracy.png")


def plot_accuracy_vs_length(df: pd.DataFrame, run_name: str, out_dir: str):
    last_step = df["step"].max()
    sub = df[df.step == last_step].copy()

    # Bin by response length
    sub["len_bin"] = pd.cut(sub["response_lengths"], bins=20)
    grouped = sub.groupby("len_bin", observed=False)["score"].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= 5]
    midpoints = [interval.mid for interval in grouped["len_bin"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    sc = ax.scatter(midpoints, grouped["mean"], c=grouped["count"],
                    cmap="viridis", s=60, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Count")
    ax.set_xlabel("Response Length (tokens)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{run_name} — Accuracy vs. Response Length (step {last_step})")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(alpha=0.3)
    save_fig(fig, out_dir, "accuracy_vs_length.png")


def plot_depth_heatmap(df: pd.DataFrame, run_name: str, out_dir: str):
    """Heatmap: steps × depth → accuracy."""
    steps = sorted(df["step"].unique())
    max_depth = min(df["depth"].max(), 15)
    depths = list(range(max_depth + 1))

    matrix = np.full((len(depths), len(steps)), np.nan)
    for j, s in enumerate(steps):
        for i, d in enumerate(depths):
            sub = df[(df.step == s) & (df.depth == d)]
            if len(sub) >= 3:
                matrix[i, j] = sub["score"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", origin="upper", cmap="RdYlGn",
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=45)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([str(d) for d in depths])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Composition Depth")
    ax.set_title(f"{run_name} — Accuracy Heatmap (step × depth)")
    save_fig(fig, out_dir, "depth_heatmap.png")


# ── Cross-run comparison ────────────────────────────────────────────────────

def plot_cross_run_accuracy(run_dfs: dict[str, pd.DataFrame], out_dir: str, is_eval: bool):
    label = "Eval" if is_eval else "Train"
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_name, df in run_dfs.items():
        steps = sorted(df["step"].unique())
        accs = [df[df.step == s]["score"].mean() for s in steps]
        ax.plot(steps, accs, "-o", label=run_name, markersize=5, linewidth=1.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Cross-Run Accuracy ({label} Rollouts)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_dir, "cross_run_accuracy.png")


def plot_cross_run_depth(run_dfs: dict[str, pd.DataFrame], out_dir: str, depth: int):
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_name, df in run_dfs.items():
        sub = df[df.depth == depth]
        steps = sorted(sub["step"].unique())
        accs = [sub[sub.step == s]["score"].mean() for s in steps]
        ax.plot(steps, accs, "-o", label=run_name, markersize=5, linewidth=1.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Cross-Run Accuracy at Depth {depth}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_dir, f"cross_run_depth_{depth}.png")


# ── Text summary ────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, run_name: str, is_eval: bool):
    label = "eval" if is_eval else "train"
    print(f"\n{'='*60}")
    print(f"  {run_name}  [{label} rollouts]")
    print(f"{'='*60}")
    steps = sorted(df["step"].unique())
    print(f"  Steps: {steps}")
    print(f"  Total entries: {len(df)}")

    print("\n  Overall accuracy per step:")
    for s in steps:
        sub = df[df.step == s]
        acc = sub["score"].mean()
        fmt = sub["has_format"].mean()
        avg_len = sub["response_lengths"].mean()
        print(f"    step {s:5d}: acc={acc:.1%}  format={fmt:.1%}  avg_len={avg_len:.0f}")

    print("\n  Accuracy by depth (final step):")
    last = df[df.step == df["step"].max()]
    for d in sorted(last["depth"].unique()):
        sub = last[last.depth == d]
        if len(sub) >= 5:
            print(f"    depth {d:2d}: {sub['score'].mean():.1%}  (n={len(sub)})")
    print()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="string-task")
    parser.add_argument("--run", default=None, help="Run name. If omitted, use --all-runs")
    parser.add_argument("--all-runs", action="store_true", help="Analyze all runs")
    parser.add_argument("--eval", action="store_true", help="Use eval rollouts")
    parser.add_argument("--out", default="plots", help="Output directory for plots")
    parser.add_argument("--cross-depth", type=int, nargs="+", default=[1, 2, 3],
                        help="Depths to compare across runs (used with --all-runs)")
    args = parser.parse_args()

    task_dir = os.path.join(BASE_DIR, args.task)

    if args.all_runs:
        runs = sorted(os.listdir(task_dir))
        run_dfs = {}
        for run in runs:
            run_dir = os.path.join(task_dir, run)
            df = load_run_data(run_dir, args.eval)
            if df.empty:
                print(f"[skip] {run}: no data")
                continue
            run_dfs[run] = df
            print_summary(df, run, args.eval)
            out = os.path.join(args.out, run)
            plot_accuracy_over_steps(df, run, out)
            plot_accuracy_by_depth(df, run, out)
            plot_response_length(df, run, out)
            plot_format_error_rate(df, run, out)
            plot_per_func_accuracy(df, run, out)
            plot_accuracy_vs_length(df, run, out)
            plot_depth_heatmap(df, run, out)

        if len(run_dfs) > 1:
            cross_out = os.path.join(args.out, "cross_run")
            print(f"\nGenerating cross-run plots → {cross_out}")
            plot_cross_run_accuracy(run_dfs, cross_out, args.eval)
            for d in args.cross_depth:
                plot_cross_run_depth(run_dfs, cross_out, d)
    else:
        if not args.run:
            parser.error("Provide --run <name> or --all-runs")
        run_dir = os.path.join(task_dir, args.run)
        df = load_run_data(run_dir, args.eval)
        if df.empty:
            print(f"[!] No data found in {run_dir}")
            return
        print_summary(df, args.run, args.eval)
        out = os.path.join(args.out, args.run)
        print(f"\nGenerating plots → {out}/")
        plot_accuracy_over_steps(df, args.run, out)
        plot_accuracy_by_depth(df, args.run, out)
        plot_response_length(df, args.run, out)
        plot_format_error_rate(df, args.run, out)
        plot_per_func_accuracy(df, args.run, out)
        plot_accuracy_vs_length(df, args.run, out)
        plot_depth_heatmap(df, args.run, out)


if __name__ == "__main__":
    main()
