#!/usr/bin/env python3
"""
Produce three plots for the response analysis section of reports/report.md:

  1. response_classification.png  — stacked area of normal/verbose/loop/q-spam
                                    over training steps, one panel per run
  2. atlimit_format_compliance.png — at-limit format compliance (%) over steps,
                                    all runs on shared axes
  3. depth_limit_gradient.png     — fraction at token limit vs composition depth,
                                    all runs overlaid with regression slopes

Usage:
    python scripts/analysis/plot_response_analysis.py
    python scripts/analysis/plot_response_analysis.py --out plots/response_analysis
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
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────

BASE_DIR    = "/data/user_data/gyeongwk/checkpoints/string-task"
TOKEN_LIMIT = 4096
Q_RUN_LEN   = 50
LOOP_NGRAM  = 40
LOOP_TAIL   = 500

RUNS = {
    "On-policy GRPO":          {"dir": "On-policy-GRPO",             "steps": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]},
    "Bootstrap GRPO":          {"dir": "Bootstrap-GRPO",             "steps": [100, 200, 300, 400, 500, 600, 661]},
    "Bootstrap REINFORCE+B":   {"dir": "Bootstrap-REINFORCE+BASELINE","steps": [100, 200, 260]},
    "Teacher GRPO":            {"dir": "Teacher-GRPO",               "steps": [100, 200, 300, 400, 500, 600]},
    "Teacher REINFORCE+B":     {"dir": "Teacher-REINFORCE+BASELINE", "steps": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1033]},
}

CAT_COLORS = {
    "normal":  "#4caf50",
    "verbose": "#ff9800",
    "loop":    "#f44336",
    "q_spam":  "#7b1fa2",
}
RUN_COLORS = ["#1f77b4", "#d62728", "#9467bd", "#ff7f0e", "#2ca02c"]

# ── data loading ──────────────────────────────────────────────────────────────

def classify(output: str, length: int) -> str:
    if length < TOKEN_LIMIT:
        return "normal"
    if re.search(r"q{" + str(Q_RUN_LEN) + r",}", output):
        return "q_spam"
    tail = output[-LOOP_TAIL:]
    if re.search(r"(` ){15,}", tail):
        return "loop"
    seen: dict[str, int] = {}
    for i in range(len(tail) - LOOP_NGRAM):
        gram = tail[i: i + LOOP_NGRAM]
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] >= 2:
            return "loop"
    return "verbose"


def get_depth(inp: str) -> int:
    m = re.search(r"return (.+)", inp)
    return len(re.findall(r"func_\d+", m.group(1))) if m else 0


def has_format(out: str) -> bool:
    return bool(re.search(r'\{"output":', out))


def load_step(run_dir: str, step: int) -> list[dict]:
    path = os.path.join(BASE_DIR, run_dir, "rollout_eval_data", f"{step}.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            e["_depth"]      = get_depth(e["input"])
            e["_at_limit"]   = e["response_lengths"] >= TOKEN_LIMIT
            e["_has_format"] = has_format(e["output"])
            e["_cat"]        = classify(e["output"], e["response_lengths"])
            rows.append(e)
    return rows


def load_run(name: str) -> dict[int, list[dict]]:
    info = RUNS[name]
    result = {}
    for step in info["steps"]:
        rows = load_step(info["dir"], step)
        if rows:
            result[step] = rows
    return result

# ── plot 1: stacked area — classification over steps ─────────────────────────

def plot_classification(out_dir: str):
    # Select the three most illustrative runs
    show_runs = ["On-policy GRPO", "Bootstrap GRPO", "Teacher REINFORCE+B"]
    cats      = ["normal", "verbose", "loop", "q_spam"]
    labels    = ["Normal", "Verbose", "Loop", "Q-spam"]

    fig, axes = plt.subplots(1, len(show_runs), figsize=(13, 4), sharey=True)

    for ax, run_name in zip(axes, show_runs):
        data = load_run(run_name)
        steps_sorted = sorted(data)
        fracs = {c: [] for c in cats}

        for step in steps_sorted:
            rows = data[step]
            n = len(rows)
            counts = {c: sum(1 for r in rows if r["_cat"] == c) for c in cats}
            for c in cats:
                fracs[c].append(counts[c] / n * 100)

        bottom = np.zeros(len(steps_sorted))
        for cat, label in zip(cats, labels):
            vals = np.array(fracs[cat])
            ax.fill_between(steps_sorted, bottom, bottom + vals,
                            label=label, color=CAT_COLORS[cat], alpha=0.85, step="mid")
            ax.step(steps_sorted, bottom + vals, color=CAT_COLORS[cat],
                    alpha=0.4, linewidth=0.8, where="mid")
            bottom += vals

        ax.set_title(run_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Training step")
        ax.set_xlim(steps_sorted[0], steps_sorted[-1])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Fraction of responses (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLORS[c], alpha=0.85) for c in cats]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Response Classification over Training", fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "response_classification.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ── plot 2: at-limit format compliance over steps ─────────────────────────────

def plot_format_compliance(out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4))

    for (run_name, color) in zip(RUNS, RUN_COLORS):
        data = load_run(run_name)
        steps, compliance = [], []
        for step in sorted(data):
            at_lim = [r for r in data[step] if r["_at_limit"]]
            if not at_lim:
                continue
            steps.append(step)
            compliance.append(np.mean([r["_has_format"] for r in at_lim]) * 100)
        if steps:
            ax.plot(steps, compliance, "-o", label=run_name, color=color,
                    linewidth=2, markersize=5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("At-limit format compliance (%)")
    ax.set_title('Fraction of token-limit responses containing {"output":...}')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "atlimit_format_compliance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ── plot 3: depth gradient — fraction at limit vs depth ───────────────────────

MAX_DEPTH = 8  # cap for depth gradient plot

def plot_depth_gradient(out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4))

    for (run_name, color) in zip(RUNS, RUN_COLORS):
        info = RUNS[run_name]
        step = info["steps"][-1]
        rows = load_step(info["dir"], step)
        if not rows:
            continue

        by_depth: dict[int, list[bool]] = defaultdict(list)
        for r in rows:
            if r["_depth"] <= MAX_DEPTH:
                by_depth[r["_depth"]].append(r["_at_limit"])

        depths, fracs = [], []
        for d in range(MAX_DEPTH + 1):
            vals = by_depth.get(d, [])
            if len(vals) >= 5:
                depths.append(d)
                fracs.append(np.mean(vals))

        if len(depths) < 2:
            continue

        slope = np.polyfit(depths, fracs, 1)[0]
        label = f"{run_name} (slope={slope:.3f}, step {step})"
        ax.plot(depths, [f * 100 for f in fracs], "-o", label=label,
                color=color, linewidth=2, markersize=5)

    ax.set_xlabel("Composition depth")
    ax.set_ylabel("% responses hitting token limit")
    ax.set_title("Depth Gradient of Limit-Hitting Probability")
    ax.set_xlim(0, MAX_DEPTH)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "depth_limit_gradient.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="plots/response_analysis")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to {args.out}/")

    plot_classification(args.out)
    plot_format_compliance(args.out)
    plot_depth_gradient(args.out)


if __name__ == "__main__":
    main()
