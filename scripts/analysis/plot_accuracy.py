"""Plot pass@1 and/or pass@n accuracy by depth across multiple experiments.

Usage:
    python scripts/plot_accuracy.py eval/exp1 eval/exp2 [--metric pass_at_1] [--output plot.png]

Each positional argument should be a directory containing accuracy.json.
The experiment label defaults to the directory name; override with --labels.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def load_accuracy(path: Path) -> dict[int, dict]:
    """Return {depth: {pass_at_1: float, pass_at_n: float}} from accuracy.json."""
    with open(path) as f:
        data = json.load(f)

    by_depth = {}
    for key, vals in data.get("by_data_source", {}).items():
        m = re.search(r"depth(\d+)", key)
        if m:
            depth = int(m.group(1))
            by_depth[depth] = vals
    return by_depth


def plot(exp_dirs: list[Path], labels: list[str], metric: str, output: str | None):
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp_dir, label in zip(exp_dirs, labels):
        accuracy_path = exp_dir / "accuracy.json"
        if not accuracy_path.exists():
            print(f"Warning: {accuracy_path} not found, skipping.")
            continue

        by_depth = load_accuracy(accuracy_path)
        depths = sorted(by_depth)
        values = [by_depth[d][metric] for d in depths]

        ax.plot(depths, values, marker="o", label=label)

    ax.set_xlabel("Depth")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Depth")
    ax.set_xticks(range(1, 9))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy by depth across experiments.")
    parser.add_argument("experiments", nargs="+", type=Path, help="Experiment directories containing accuracy.json")
    parser.add_argument(
        "--metric",
        choices=["pass_at_1", "pass_at_n"],
        default="pass_at_1",
        help="Which metric to plot (default: pass_at_1)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Display labels for each experiment (defaults to directory name)",
    )
    parser.add_argument("--output", "-o", help="Save figure to this path instead of showing it")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.experiments):
        parser.error("--labels must have the same number of entries as experiments")

    labels = args.labels if args.labels else [p.name for p in args.experiments]
    plot(args.experiments, labels, args.metric, args.output)


if __name__ == "__main__":
    main()
