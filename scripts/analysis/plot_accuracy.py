"""Plot accuracy by level from a CSV results file.

Usage:
    # Plot all rows
    python scripts/analysis/plot_accuracy.py results/main.csv

    # Filter to a specific data group
    python scripts/analysis/plot_accuracy.py results/main.csv --data On-policy
    python scripts/analysis/plot_accuracy.py results/main.csv --data Bootstrap

    # Save to file
    python scripts/analysis/plot_accuracy.py results/main.csv --output plot.png

CSV format expected:
    Data, Loss Function, Level 1, Level 2, ..., Level 8, ...
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

BASE_FONT_SIZE = 14

LEVEL_COLUMNS = [f"Level {i}" for i in range(1, 9)]


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


def load_csv(path: Path, data_group: str | None) -> list[dict]:
    """Return rows (optionally filtered by data_group) as list of {label, levels, values}."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        has_name_col = "Name" in headers
        for row in reader:
            if data_group is not None and row["Data"].strip() != data_group:
                continue
            if has_name_col and row.get("Name", "").strip():
                label = row["Name"].strip()
            elif data_group is None:
                label = f"{row['Data'].strip()} - {row['Loss Function'].strip()}"
            else:
                label = row["Loss Function"].strip()
            levels, values = [], []
            for col in LEVEL_COLUMNS:
                val = row.get(col, "").strip()
                if val != "":
                    levels.append(int(col.split()[1]))
                    values.append(float(val))
            if levels:
                rows.append({"label": label, "levels": levels, "values": values})
    return rows


def plot(csv_path: Path, data_group: str | None, output: str | None):
    configure_plot_style()
    rows = load_csv(csv_path, data_group)

    if not rows:
        msg = f"data group '{data_group}'" if data_group else "any rows"
        print(f"No rows found for {msg} in {csv_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        ax.plot(row["levels"], row["values"], marker="o", label=row["label"])

    ax.set_xlabel("Level")
    ax.set_ylabel("Accuracy")
    title = f"Accuracy by Level — {data_group}" if data_group else "Accuracy by Level"
    ax.set_title(title)
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
    parser = argparse.ArgumentParser(description="Plot accuracy by level from a CSV results file.")
    parser.add_argument("csv", type=Path, help="Path to results CSV file")
    parser.add_argument(
        "--data",
        default=None,
        help="Filter to a specific data group (e.g. On-policy, Bootstrap). Omit to plot all rows.",
    )
    parser.add_argument("--output", "-o", help="Save figure to this path instead of showing it")
    args = parser.parse_args()

    plot(args.csv, args.data, args.output)


if __name__ == "__main__":
    main()
