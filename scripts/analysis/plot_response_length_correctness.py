"""Compare response token lengths for correct vs incorrect responses across eval parquet files.

Usage:
    python scripts/analysis/plot_response_length_correctness.py \\
        results/eval/grpo/rollout.parquet \\
        results/eval/sft-onpolicy/rollout.parquet \\
        --labels GRPO SFT \\
        --output results/plots/response_length_correctness.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verl.utils.reward_score.codeio import extract_last_complete_json, is_close

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


def is_correct(response: str, ground_truth: dict) -> bool:
    parsed = extract_last_complete_json(response)
    if parsed is None or "output" not in parsed:
        return False
    return bool(is_close(parsed["output"], ground_truth["ref_output"]))


def compute_lengths_by_correctness(parquet_path: Path, tokenizer, response_key: str
                                   ) -> tuple[list[int], list[int]]:
    df = pd.read_parquet(parquet_path)

    if response_key not in df.columns:
        response_key = "responses" if "responses" in df.columns else "response"

    correct_lengths, incorrect_lengths = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=parquet_path.parent.name, leave=False):
        # Get response text (first response only)
        val = row[response_key]
        if isinstance(val, (list, np.ndarray)):
            text = str(val[0]) if len(val) > 0 else ""
        else:
            text = str(val) if pd.notna(val) else ""

        # Determine correctness
        gt = json.loads(row["reward_model"]["ground_truth"])
        correct = is_correct(text, gt)

        # Compute token length
        enc = tokenizer(text, return_tensors=None, add_special_tokens=False)
        length = len(enc["input_ids"])

        if correct:
            correct_lengths.append(length)
        else:
            incorrect_lengths.append(length)

    return correct_lengths, incorrect_lengths


def make_bxp_stat(lengths: list[int], label: str) -> dict:
    a = np.array(lengths)
    return {
        "med": float(np.median(a)),
        "q1": float(np.percentile(a, 25)),
        "q3": float(np.percentile(a, 75)),
        "whislo": float(a.min()),
        "whishi": float(a.max()),
        "label": label,
    }


def plot(parquet_paths: list[Path], labels: list[str], tokenizer_name: str,
         response_key: str, output: str | None):
    configure_plot_style()

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    all_correct, all_incorrect = [], []
    for path, label in zip(parquet_paths, labels):
        correct, incorrect = compute_lengths_by_correctness(path, tokenizer, response_key)
        all_correct.append(correct)
        all_incorrect.append(incorrect)
        print(
            f"{label}: correct n={len(correct)} median={np.median(correct):.0f} | "
            f"incorrect n={len(incorrect)} median={np.median(incorrect):.0f}"
        )

    # Build interleaved bxp stats: [correct_0, incorrect_0, correct_1, incorrect_1, ...]
    bxp_stats = []
    ytick_positions = []
    ytick_labels = []
    group_gap = 1.0   # gap between methods
    box_gap = 0.5     # gap between correct/incorrect within a method

    pos = 0.0
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    box_colors = []   # (color, is_correct) per box

    for i, label in enumerate(labels):
        c_pos = pos
        i_pos = pos + box_gap
        bxp_stats.append({**make_bxp_stat(all_correct[i], ""), "pos": c_pos})
        bxp_stats.append({**make_bxp_stat(all_incorrect[i], ""), "pos": i_pos})
        box_colors.append((colors[i % len(colors)], True))
        box_colors.append((colors[i % len(colors)], False))
        ytick_positions.append((c_pos + i_pos) / 2)
        ytick_labels.append(label)
        pos += box_gap + group_gap

    # Strip "pos" out for bxp (it doesn't accept it as a key)
    positions = [s.pop("pos") for s in bxp_stats]

    fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 1.2)))

    bp = ax.bxp(bxp_stats, positions=positions, vert=False, patch_artist=True,
                showfliers=False, widths=0.35,
                medianprops={"linewidth": 2, "color": "white"})

    for patch, (color, is_corr) in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85 if is_corr else 0.4)
        if not is_corr:
            patch.set_hatch("//")

    for element in ("whiskers", "caps"):
        items = bp[element]
        for j, line in enumerate(items):
            color, _ = box_colors[j // 2]
            line.set_color(color)

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)
    ax.set_xscale("log")
    ax.set_xlabel("Response Length (tokens, log scale)")
    ax.set_title("Response Length: Correct vs Incorrect\n(box = p25–p75, line = median, whiskers = min–max)")
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="grey", alpha=0.85, label="Correct"),
        Patch(facecolor="grey", alpha=0.4, hatch="//", label="Incorrect"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.8)

    fig.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare response lengths for correct vs incorrect responses."
    )
    parser.add_argument("parquet_files", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+",
                        help="Label per file (defaults to parent directory name)")
    parser.add_argument("--tokenizer", default="meta-llama/llama-3.1-8b")
    parser.add_argument("--response-key", default="response")
    parser.add_argument("--output", "-o")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.parquet_files):
        parser.error("--labels count must match number of files")

    labels = args.labels if args.labels else [p.parent.name for p in args.parquet_files]
    plot(args.parquet_files, labels, args.tokenizer, args.response_key, args.output)


if __name__ == "__main__":
    main()
