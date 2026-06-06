#!/usr/bin/env python3
"""
Classify rollout responses into four categories per training step:
  normal   — response ends before the token limit
  verbose  — hits limit but no detectable repetition
  loop     — hits limit with a repeated phrase or backtick sequence in the tail
  q_spam   — hits limit with a 50+ consecutive-q run in the output field

Usage:
    python scripts/analysis/classify_rollouts.py
    python scripts/analysis/classify_rollouts.py --run Bootstrap-GRPO
    python scripts/analysis/classify_rollouts.py --run Bootstrap-GRPO --step 300
    python scripts/analysis/classify_rollouts.py --token-limit 8192

Classification rules (applied in priority order):
  1. normal  : response_lengths < TOKEN_LIMIT
  2. q_spam  : hits limit AND re.search(r'q{50,}', output)
  3. loop    : hits limit AND (backtick-space run of 15+ tokens in tail OR
                              any 40-char substring appears 2+ times in last 500 chars)
  4. verbose : everything else that hit the limit

Notes on agreement with the rollout_degradation_analysis.md report:
  - 'normal' and 'q_spam' match the report exactly (they are algorithmic).
  - Late-step loop counts (when degradation is severe) agree to within ~1%.
  - Early-step loop/verbose split differs: the report's agent classified
    computation-expansion loops (e.g. `ndssfovq[1:]+n = ndssfovq[1:]+n...`)
    as 'verbose', whereas this script correctly flags them as 'loop'.
    Those early-step verbose percentages in the report were manual estimates.
"""

import argparse
import json
import os
import re
from collections import Counter

BASE_DIR = "/data/user_data/gyeongwk/checkpoints"

TOKEN_LIMIT = 4096
Q_RUN_LEN = 50       # consecutive q's → q_spam
LOOP_NGRAM = 40      # min substring length for repetition check
LOOP_MIN_COUNT = 2   # how many times it must appear
LOOP_TAIL = 500      # how many chars from the end to examine

ALL_RUNS = [
    "On-policy-GRPO",
    "Bootstrap-GRPO",
    "Bootstrap-REINFORCE+BASELINE",
    "Teacher-GRPO",
    "Teacher-REINFORCE+BASELINE",
]

CATEGORIES = ["normal", "verbose", "loop", "q_spam"]


def classify(output: str, length: int, token_limit: int = TOKEN_LIMIT) -> str:
    if length < token_limit:
        return "normal"

    if re.search(r"q{" + str(Q_RUN_LEN) + r",}", output):
        return "q_spam"

    tail = output[-LOOP_TAIL:]

    if re.search(r"(` ){15,}", tail):
        return "loop"

    seen: dict[str, int] = {}
    for i in range(len(tail) - LOOP_NGRAM):
        gram = tail[i : i + LOOP_NGRAM]
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] >= LOOP_MIN_COUNT:
            return "loop"

    return "verbose"


def load_steps(run_dir: str, step_filter: int | None, token_limit: int = TOKEN_LIMIT) -> dict[int, list[dict]]:
    data_dir = os.path.join(run_dir, "rollout_eval_data")
    if not os.path.isdir(data_dir):
        return {}

    steps: dict[int, list[dict]] = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl") or fname == "tmp.jsonl":
            continue
        step = int(fname.replace(".jsonl", ""))
        if step_filter is not None and step != step_filter:
            continue
        entries = []
        with open(os.path.join(data_dir, fname)) as f:
            for line in f:
                entries.append(json.loads(line))
        steps[step] = entries

    return steps


def print_table(run_name: str, steps: dict[int, list[dict]], token_limit: int = TOKEN_LIMIT):
    print(f"\n### {run_name}")
    print(f"| {'Step':>5} | {'Normal':>7} | {'Verbose':>8} | {'Loop':>5} | {'q-Spam':>7} | {'N':>5} |")
    print(f"|{'------':->6}-|{'--------':->8}-|{'---------':->9}-|{'------':->6}-|{'--------':->8}-|{'------':->6}-|")

    for step in sorted(steps):
        entries = steps[step]
        cats = [classify(e["output"], e["response_lengths"], token_limit) for e in entries]
        counts = Counter(cats)
        n = len(cats)
        normal  = counts["normal"]  / n
        verbose = counts["verbose"] / n
        loop    = counts["loop"]    / n
        q_spam  = counts["q_spam"]  / n
        print(
            f"| {step:>5} | {normal:>6.0%} | {verbose:>7.0%} | {loop:>4.0%} | {q_spam:>6.0%} | {n:>5} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Classify rollout responses by failure mode")
    parser.add_argument("--task", default="string-task")
    parser.add_argument("--run", default=None, help="Single run name; omit for all runs")
    parser.add_argument("--step", type=int, default=None, help="Single step; omit for all steps")
    parser.add_argument("--token-limit", type=int, default=TOKEN_LIMIT,
                        help=f"Context limit used during generation (default: {TOKEN_LIMIT})")
    args = parser.parse_args()

    token_limit = args.token_limit

    task_dir = os.path.join(BASE_DIR, args.task)
    runs = [args.run] if args.run else ALL_RUNS

    for run in runs:
        run_dir = os.path.join(task_dir, run)
        if not os.path.isdir(run_dir):
            print(f"[skip] {run}: directory not found")
            continue

        steps = load_steps(run_dir, args.step)
        if not steps:
            print(f"[skip] {run}: no eval rollout data found")
            continue

        print_table(run, steps, token_limit)

    print()


if __name__ == "__main__":
    main()
