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

SMOOTHING = 0.9

RUN_LABELS = {
    "math-onpolicy-GRPO-Qwen3-1.7B": "GRPO",
    "math-onpolicy-PosNeg-Qwen3-1.7B": "PosNeg",
    "math-onpolicy-Reinforce-Baseline-Qwen3-1.7B": "Reinforce-Baseline",
    "math-onpolicy-SFT-Qwen3-1.7B": "SFT",
}

# math-onpolicy-SFT-Qwen3-1.7B's wandb logging genuinely stops at step 1198 (job
# ended once response length collapsed to ~0 -- see math_task_response_length_dynamics.md
# section 5); other runs continue to ~1900. Not a rendering bug -- rather than leaving
# a truncated line (which reads as a plotting glitch), we extend it flat at its last
# logged value out to the other runs' range, dashed to mark it as inferred/not logged.
SFT_RUN_NAME = "math-onpolicy-SFT-Qwen3-1.7B"

# Font sizes are tuned per-panel for how the figure is actually placed in the
# paper: the three pass@1 panels sit 3-across at 0.3\textwidth each (~1.65in
# printed), so they need a much larger native font than a full-width figure to
# stay legible after LaTeX shrinks them. figsize is chosen so BASE_FONT_SIZE
# lands around 9-10pt effective at print size.
PANELS = [
    ("math_qwen3_1.7b_response_length.csv", "Avg response length (tokens)", "math_qwen3_response_length.png", True, (10, 6), 14, True),
    ("math_qwen3_1.7b_pass1_easy.csv", "Pass@1 (easy)", "math_qwen3_pass1_easy.png", False, (3.6, 3.3), 19, False),
    ("math_qwen3_1.7b_pass1_medium.csv", "Pass@1 (medium)", "math_qwen3_pass1_medium.png", False, (3.6, 3.3), 19, False),
    ("math_qwen3_1.7b_pass1_hard.csv", "Pass@1 (hard)", "math_qwen3_pass1_hard.png", False, (3.6, 3.3), 19, False),
]


def configure_plot_style(base_font_size: int):
    plt.rcParams.update(
        {
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 3,
            "axes.labelsize": base_font_size + 1,
            "xtick.labelsize": base_font_size,
            "ytick.labelsize": base_font_size,
            "legend.fontsize": base_font_size - 4,
            "figure.titlesize": base_font_size + 3,
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


def plot_metric(results_dir: str, fname: str, ylabel: str, output_path: str, apply_smoothing: bool,
                 figsize: tuple[float, float], base_font_size: int, extend_sft_end: bool):
    configure_plot_style(base_font_size)
    fig, ax = plt.subplots(figsize=figsize)

    series = load_wandb_csv(str(Path(results_dir) / fname))
    max_step = max((s for run in series.values() for s, _ in run), default=0)

    for run, label in RUN_LABELS.items():
        if run not in series or not series[run]:
            continue
        xs = [s for s, _ in series[run]]
        ys = [v for _, v in series[run]]
        if apply_smoothing:
            ys = smooth(ys, SMOOTHING)
        (line,) = ax.plot(xs, ys, linewidth=1.8, label=label)
        if extend_sft_end and run == SFT_RUN_NAME and xs and xs[-1] < max_step:
            # SFT's wandb job stopped logging early (collapsed to ~0 response length,
            # see math_task_response_length_dynamics.md sec 5). Extend flat at its last
            # value so the line doesn't look cut off; dashed to mark it as inferred,
            # not actually logged.
            ax.plot([xs[-1], max_step], [ys[-1], ys[-1]], linewidth=1.8, linestyle="--",
                     color=line.get_color())

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Training step")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=base_font_size - 6, loc="best", framealpha=0.9)

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

    for fname, ylabel, out_name, apply_smoothing, figsize, base_font_size, extend_sft_end in PANELS:
        plot_metric(args.results_dir, fname, ylabel, str(Path(args.output_dir) / out_name),
                    apply_smoothing, figsize, base_font_size, extend_sft_end)


if __name__ == "__main__":
    main()
