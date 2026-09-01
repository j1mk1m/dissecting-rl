"""Plot response length and pass@1 (easy/medium/hard) trajectories for the
on-policy math-task Qwen3-1.7B runs (GRPO, PosNeg, Reinforce-Baseline, SFT),
directly from wandb CSV exports (no need to re-parse raw rollout JSONL
files). Each metric is saved as its own figure with a legend.

Source CSVs (results/):
  - math_qwen3_1.7b_response_length.csv  (rollout/avg_response_length)
  - math_qwen3_1.7b_pass1_easy.csv       (val-core/math-easy/reward/pass@1)
  - math_qwen3_1.7b_pass1_medium.csv     (val-core/math-medium/reward/pass@1)
  - math_qwen3_1.7b_pass1_hard.csv       (val-core/math-hard/reward/pass@1)

Each wandb export has one "Name: <run> - <metric>" column per run plus
__MIN/__MAX shadow columns (dropped here).

Usage:
    python scripts/analysis/plot_math_qwen3_wandb_metrics.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

BASE_FONT_SIZE = 14
SMOOTHING = 0.9

RUN_LABELS = {
    "math-onpolicy-GRPO-Qwen3-1.7B": "GRPO",
    "math-onpolicy-PosNeg-Qwen3-1.7B": "PosNeg",
    "math-onpolicy-Reinforce-Baseline-Qwen3-1.7B": "Reinforce-Baseline",
    "math-onpolicy-SFT-Qwen3-1.7B": "SFT",
}

PANELS = [
    ("math_qwen3_1.7b_response_length.csv", "Avg response length (tokens)", "math_qwen3_response_length.png", True),
    ("math_qwen3_1.7b_pass1_easy.csv", "Pass@1 (math-easy)", "math_qwen3_pass1_easy.png", False),
    ("math_qwen3_1.7b_pass1_medium.csv", "Pass@1 (math-medium)", "math_qwen3_pass1_medium.png", False),
    ("math_qwen3_1.7b_pass1_hard.csv", "Pass@1 (math-hard)", "math_qwen3_pass1_hard.png", False),
]


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE + 3,
            "axes.labelsize": BASE_FONT_SIZE + 1,
            "xtick.labelsize": BASE_FONT_SIZE,
            "ytick.labelsize": BASE_FONT_SIZE,
            "legend.fontsize": BASE_FONT_SIZE,
            "figure.titlesize": BASE_FONT_SIZE + 3,
        }
    )


def smooth(values: list[float], weight: float) -> list[float]:
    """Exponential moving average, matching plot_response_length.py's smooth()."""
    result, last = [], None
    for v in values:
        last = v if last is None else last * weight + v * (1 - weight)
        result.append(last)
    return result


def load_wandb_csv(path: str):
    """Return {run_name: [(step, value), ...]} from a wandb CSV export,
    dropping the __MIN/__MAX shadow columns."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = {}
        for i, h in enumerate(header):
            if i == 0 or h.endswith("__MIN") or h.endswith("__MAX"):
                continue
            run = h.split("Name: ", 1)[1].rsplit(" - ", 1)[0]
            cols[i] = run

        series = {run: [] for run in cols.values()}
        for row in reader:
            step = int(row[0])
            for i, run in cols.items():
                val = row[i]
                if val == "":
                    continue
                series[run].append((step, float(val)))
    for run in series:
        series[run].sort()
    return series


def plot_metric(results_dir: str, fname: str, ylabel: str, output_path: str, apply_smoothing: bool):
    fig, ax = plt.subplots(figsize=(10, 6))

    series = load_wandb_csv(str(Path(results_dir) / fname))
    for run, label in RUN_LABELS.items():
        if run not in series or not series[run]:
            continue
        xs = [s for s, _ in series[run]]
        ys = [v for _, v in series[run]]
        if apply_smoothing:
            ys = smooth(ys, SMOOTHING)
        ax.plot(xs, ys, linewidth=1.8, label=label)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Training step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()

    configure_plot_style()
    for fname, ylabel, out_name, apply_smoothing in PANELS:
        plot_metric(args.results_dir, fname, ylabel, str(Path(args.output_dir) / out_name), apply_smoothing)


if __name__ == "__main__":
    main()
