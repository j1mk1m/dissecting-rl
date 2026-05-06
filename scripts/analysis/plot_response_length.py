"""Plot average response length over training steps from a W&B-exported CSV.

Usage:
    python scripts/analysis/plot_response_length.py results/response_length.csv
    python scripts/analysis/plot_response_length.py results/response_length.csv --output results/plots/response_length.png
    python scripts/analysis/plot_response_length.py results/response_length.csv --no-band

CSV format expected (W&B export):
    Step, "Name: <method> - rollout/avg_response_length",
          "Name: <method> - rollout/avg_response_length__MIN",
          "Name: <method> - rollout/avg_response_length__MAX", ...
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

BASE_FONT_SIZE = 14
ALPHA_BAND = 0.15
X_MAX = 1100
X_TICK_STEP = 200


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


def parse_methods(headers: list[str]) -> dict[str, dict[str, str]]:
    """Return {method_label: {mean: col, min: col, max: col}} from CSV headers."""
    methods = {}
    for h in headers:
        # Match mean column (no __MIN / __MAX suffix)
        m = re.match(r"Name:\s*(.+?)\s*-\s*rollout/avg_response_length$", h)
        if m:
            # Strip "On-policy-" prefix for cleaner labels
            label = re.sub(r"^On-policy-", "", m.group(1).strip())
            methods[label] = {"mean": h, "min": None, "max": None}

    for h in headers:
        m = re.match(r"Name:\s*(.+?)\s*-\s*rollout/avg_response_length__(MIN|MAX)$", h)
        if m:
            label = re.sub(r"^On-policy-", "", m.group(1).strip())
            kind = m.group(2).lower()
            if label in methods:
                methods[label][kind] = h

    return methods


def load_csv(path: Path) -> tuple[list[int], dict[str, dict]]:
    """Return (steps, {method: {mean, min, max: list[float]}})."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return [], {}

    methods = parse_methods(list(rows[0].keys()))
    steps = []
    series: dict[str, dict[str, list]] = {m: {"mean": [], "min": [], "max": []} for m in methods}

    for row in rows:
        step_val = row.get("Step", "").strip()
        if not step_val:
            continue
        steps.append(int(step_val))
        for label, cols in methods.items():
            for kind in ("mean", "min", "max"):
                col = cols[kind]
                val = row.get(col, "").strip() if col else ""
                series[label][kind].append(float(val) if val else None)

    return steps, series


def smooth(values: list[float | None], weight: float) -> list[float]:
    """Exponential moving average; None values are skipped."""
    result, last = [], None
    for v in values:
        if v is None:
            result.append(float("nan"))
            continue
        last = v if last is None else last * weight + v * (1 - weight)
        result.append(last)
    return result


def plot(csv_path: Path, show_band: bool, smoothing: float, output: str | None):
    configure_plot_style()
    steps, series = load_csv(csv_path)

    if not series:
        print(f"No data found in {csv_path}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, data in series.items():
        valid = [(s, m, lo, hi) for s, m, lo, hi in zip(steps, data["mean"], data["min"], data["max"])
                 if m is not None and s <= X_MAX]
        if not valid:
            continue
        xs, means, mins, maxs = zip(*valid)

        smoothed = smooth(list(means), smoothing)
        (line,) = ax.plot(xs, smoothed, marker="", linewidth=1.8, label=label)

        if show_band and any(lo is not None for lo in mins):
            lo_vals = smooth([lo if lo is not None else m for lo, m in zip(mins, means)], smoothing)
            hi_vals = smooth([hi if hi is not None else m for hi, m in zip(maxs, means)], smoothing)
            ax.fill_between(xs, lo_vals, hi_vals, alpha=ALPHA_BAND, color=line.get_color())

    ax.set_xlabel("Training Step", fontsize=BASE_FONT_SIZE + 2)
    ax.set_ylabel("Avg Response Length (tokens)", fontsize=BASE_FONT_SIZE + 2)
    ax.set_xlim(0, X_MAX)
    ax.set_xticks(range(0, X_MAX + 1, X_TICK_STEP))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot response length over training steps from a W&B CSV export.")
    parser.add_argument("csv", type=Path, help="Path to response_length CSV file")
    parser.add_argument("--output", "-o", help="Save figure to this path instead of showing it")
    parser.add_argument("--band", action="store_true", default=False,
                        help="Show min/max shaded band around the mean")
    parser.add_argument("--smoothing", type=float, default=0.9, metavar="W",
                        help="EMA smoothing weight 0–1 (default: 0.9; 0 = no smoothing)")
    args = parser.parse_args()

    plot(args.csv, args.band, args.smoothing, args.output)


if __name__ == "__main__":
    main()
