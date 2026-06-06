#!/usr/bin/env python3
"""
Browse rollout examples interactively.

Usage:
    python scripts/analysis/browse_rollouts.py --run Bootstrap-GRPO
    python scripts/analysis/browse_rollouts.py --run Bootstrap-GRPO --step 300 --correct
    python scripts/analysis/browse_rollouts.py --run Bootstrap-GRPO --step 300 --wrong --depth 1 2 3
    python scripts/analysis/browse_rollouts.py --run Bootstrap-GRPO --eval --step 100 --n 5
"""

import argparse
import json
import os
import random
import re
import sys
import textwrap

BASE_DIR = "/data/user_data/gyeongwk/checkpoints"


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


def extract_predicted_answer(output: str) -> str | None:
    m = re.search(r'\{"output":\s*"(.*?)"\}', output, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r'\{"output":\s*([^}]+)\}', output)
    if m:
        return m.group(1).strip().strip('"')
    return None


def has_valid_format(output: str) -> bool:
    return bool(re.search(r'\{"output":', output))


def list_runs(task: str) -> list[str]:
    task_dir = os.path.join(BASE_DIR, task)
    if not os.path.isdir(task_dir):
        return []
    return sorted(os.listdir(task_dir))


def load_entries(run_dir: str, step: int | None, is_eval: bool) -> list[dict]:
    subdir = "rollout_eval_data" if is_eval else "rollout_data"
    data_dir = os.path.join(run_dir, subdir)
    if not os.path.isdir(data_dir):
        print(f"[!] Directory not found: {data_dir}")
        return []

    files = sorted(
        (f for f in os.listdir(data_dir) if f.endswith(".jsonl") and f != "tmp.jsonl"),
        key=lambda f: int(f.replace(".jsonl", "")),
    )

    if step is not None:
        target = f"{step}.jsonl"
        if target not in files:
            available = [f.replace(".jsonl", "") for f in files]
            print(f"[!] Step {step} not found. Available: {available}")
            return []
        files = [target]

    entries = []
    for fname in files:
        with open(os.path.join(data_dir, fname)) as f:
            for line in f:
                entries.append(json.loads(line))
    return entries


def format_entry(entry: dict, idx: int, total: int, show_full: bool = False) -> str:
    try:
        width = min(os.get_terminal_size().columns, 100)
    except OSError:
        width = 100
    sep = "─" * width

    depth = get_composition_depth(entry["input"])
    funcs = get_func_ids(entry["input"])
    correct = entry["score"] == 1.0
    has_fmt = has_valid_format(entry["output"])
    predicted = extract_predicted_answer(entry["output"])

    # Extract the code snippet from input
    code_m = re.search(r"def main_solution.*?(?=Can you)", entry["input"], re.DOTALL)
    code = code_m.group(0).strip() if code_m else "(no code found)"

    status = "CORRECT" if correct else "WRONG"
    status_color = "\033[92m" if correct else "\033[91m"
    reset = "\033[0m"

    lines = [
        sep,
        f"Example {idx + 1}/{total}  |  Step: {entry.get('step', '?')}  |  "
        f"{status_color}{status}{reset}  |  Depth: {depth}  |  Funcs: {funcs}  |  "
        f"Format OK: {has_fmt}  |  Tokens: {entry['response_lengths']}",
        sep,
        "[CODE]",
        code,
        "",
        f"[PREDICTED ANSWER]  {predicted!r}",
        "",
    ]

    reasoning = entry["output"]
    if not show_full and len(reasoning) > 800:
        reasoning = reasoning[:800] + "\n... (truncated, use --full to see all)"

    lines.append("[REASONING]")
    lines.append(reasoning)
    lines.append(sep)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Browse rollout examples")
    parser.add_argument("--task", default="string-task", help="Task name (default: string-task)")
    parser.add_argument("--run", required=True, help="Run name, e.g. Bootstrap-GRPO")
    parser.add_argument("--step", type=int, default=None, help="Training step (default: latest)")
    parser.add_argument("--eval", action="store_true", help="Use eval rollouts (default: train)")
    parser.add_argument("--correct", action="store_true", help="Show only correct examples")
    parser.add_argument("--wrong", action="store_true", help="Show only wrong examples")
    parser.add_argument("--no-format", action="store_true", help="Show only examples with bad format")
    parser.add_argument("--depth", type=int, nargs="+", default=None, help="Filter by composition depth(s)")
    parser.add_argument("--func", type=int, nargs="+", default=None, help="Filter by func id(s) present in composition")
    parser.add_argument("-n", type=int, default=10, help="Number of examples to show (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Show full reasoning (not truncated)")
    parser.add_argument("--list-runs", action="store_true", help="List available runs and exit")
    args = parser.parse_args()

    if args.list_runs:
        runs = list_runs(args.task)
        print(f"Available runs in {args.task}:")
        for r in runs:
            print(f"  {r}")
        return

    run_dir = os.path.join(BASE_DIR, args.task, args.run)
    if not os.path.isdir(run_dir):
        print(f"[!] Run not found: {run_dir}")
        print(f"Available: {list_runs(args.task)}")
        sys.exit(1)

    entries = load_entries(run_dir, args.step, args.eval)
    if not entries:
        sys.exit(1)

    # Annotate entries
    for e in entries:
        e["_depth"] = get_composition_depth(e["input"])
        e["_funcs"] = get_func_ids(e["input"])
        e["_has_format"] = has_valid_format(e["output"])

    # Apply filters
    filtered = entries
    if args.correct:
        filtered = [e for e in filtered if e["score"] == 1.0]
    elif args.wrong:
        filtered = [e for e in filtered if e["score"] != 1.0]
    if args.no_format:
        filtered = [e for e in filtered if not e["_has_format"]]
    if args.depth is not None:
        filtered = [e for e in filtered if e["_depth"] in args.depth]
    if args.func is not None:
        filtered = [e for e in filtered if any(f in e["_funcs"] for f in args.func)]

    print(f"\nLoaded {len(entries)} entries → {len(filtered)} after filters")

    if not filtered:
        print("[!] No entries match the filters.")
        return

    random.seed(args.seed)
    sample = random.sample(filtered, min(args.n, len(filtered)))

    for i, entry in enumerate(sample):
        print(format_entry(entry, i, len(sample), show_full=args.full))
        if i < len(sample) - 1:
            try:
                input("\nPress Enter for next, Ctrl+C to stop...")
            except (KeyboardInterrupt, EOFError):
                print("\nStopped.")
                break


if __name__ == "__main__":
    main()
