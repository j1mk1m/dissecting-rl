"""Compare response token-length statistics (min, mean, max) across multiple parquet files.

Usage:
    python scripts/analysis/plot_response_length_eval.py \\
        results/eval/grpo/rollout.parquet \\
        results/eval/sft-onpolicy/rollout.parquet \\
        --labels GRPO SFT \\
        --output results/plots/response_length_eval.png

Each bar shows the mean; error bars span min to max.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

BASE_FONT_SIZE = 12


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


def _is_present(val) -> bool:
    if isinstance(val, np.ndarray):
        return bool(np.any(pd.notna(val)))
    return bool(pd.notna(val))


def _cell_to_strings(val) -> list[str]:
    if isinstance(val, (list, np.ndarray)):
        strings = []
        for r in (val if isinstance(val, list) else val.tolist()):
            if isinstance(r, (list, np.ndarray)):
                strings.extend(_cell_to_strings(r))
            else:
                strings.append(str(r) if _is_present(r) else "")
        return strings
    return [str(val) if _is_present(val) else ""]


def get_response_strings(df: pd.DataFrame, response_key: str = "response") -> list[str]:
    if response_key not in df.columns:
        if "responses" in df.columns:
            response_key = "responses"
        else:
            raise KeyError(f"No response column found. Columns: {list(df.columns)}")
    out = []
    for val in df[response_key]:
        strings = _cell_to_strings(val)
        out.append(strings[0] if strings else "")
    return out


def compute_stats(parquet_path: Path, tokenizer, response_key: str) -> dict:
    df = pd.read_parquet(parquet_path)
    texts = get_response_strings(df, response_key)
    lengths = []
    for t in tqdm(texts, desc=str(parquet_path.parent.name), leave=False):
        enc = tokenizer(t, return_tensors=None, add_special_tokens=False)
        lengths.append(len(enc["input_ids"]))
    lengths = np.array(lengths)
    return {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "min": float(lengths.min()),
        "max": float(lengths.max()),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
        "std": float(lengths.std()),
        "n": len(lengths),
    }


def plot(parquet_paths: list[Path], labels: list[str], tokenizer_name: str,
         response_key: str, output: str | None):
    configure_plot_style()

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    stats = []
    for path in parquet_paths:
        s = compute_stats(path, tokenizer, response_key)
        stats.append(s)
        print(f"{path}: mean={s['mean']:.0f}, median={s['median']:.0f}, p25={s['p25']:.0f}, p75={s['p75']:.0f}, min={s['min']:.0f}, max={s['max']:.0f}")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    bxp_stats = [
        {
            "med": s["median"],
            "q1": s["p25"],
            "q3": s["p75"],
            "whislo": s["min"],
            "whishi": s["max"],
            "label": label,
        }
        for s, label in zip(stats, labels)
    ]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.9)))

    bp = ax.bxp(bxp_stats, vert=False, patch_artist=True, showfliers=False,
                medianprops={"linewidth": 2, "color": "white"})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ("whiskers", "caps"):
        for line, color in zip(
            [bp[element][j] for j in range(len(bp[element]))],
            [c for c in colors for _ in range(2)],
        ):
            line.set_color(color)

    ax.set_xscale("log")
    ax.set_xlabel("Response Length (tokens, log scale)")
    ax.set_title("Response Length at Evaluation\n(box = p25–p75, whiskers = min–max)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare response token-length stats across parquet eval files."
    )
    parser.add_argument("parquet_files", nargs="+", type=Path,
                        help="Parquet files to compare")
    parser.add_argument("--labels", nargs="+",
                        help="Display label for each file (defaults to parent directory name)")
    parser.add_argument("--tokenizer", default="meta-llama/llama-3.1-8b",
                        help="HuggingFace tokenizer (default: meta-llama/llama-3.1-8b)")
    parser.add_argument("--response-key", default="response",
                        help="Column name for responses (default: response)")
    parser.add_argument("--output", "-o",
                        help="Save figure to this path instead of showing it")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.parquet_files):
        parser.error("--labels must have the same number of entries as parquet files")

    labels = args.labels if args.labels else [p.parent.name for p in args.parquet_files]
    plot(args.parquet_files, labels, args.tokenizer, args.response_key, args.output)


if __name__ == "__main__":
    main()
