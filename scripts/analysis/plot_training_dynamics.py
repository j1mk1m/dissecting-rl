"""Generate the missing paper figures from results/*.csv (see
fetch_wandb_training_dynamics.py and results/main.csv / results/baseline.csv).

Produces, into reports/imgs/:
    data_vary_loss_fn_sft_no_onpolicy.png  -- Mimic Zone: SFT accuracy by level, data axis
    loss_fn_on_policy.png                  -- Generalization Zone: accuracy by level, loss axis
    response_length.png                    -- accuracy + response length over steps, on-policy
    entropy.png                            -- eval entropy (depth-2) over steps, on-policy
    grad_norm.png                          -- actor/grad_norm over steps, on-policy
    grad_norm_collapse.png                 -- on-policy vs. off-policy grad norm (log scale)
    math_bootstrap_sft_pass1.png           -- math Bootstrap-SFT-easy: easy/med/hard pass@1

Usage:
    python scripts/analysis/plot_training_dynamics.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
OUT_DIR = Path("reports/imgs")
BASE_FONT_SIZE = 14

# Fixed categorical color assignment (dataviz skill default palette, slots
# assigned by role and held constant across every figure).
COLOR = {
    "Base": "#6b6b66",  # neutral gray, not a categorical slot -- reference line
    "SFT": "#2a78d6",  # slot 1 blue
    "Bootstrap": "#1baf7a",  # slot 2 aqua
    "POS+NEG": "#eda100",  # slot 3 yellow
    "On-policy": "#008300",  # slot 4 green
    "REINFORCE+Baseline": "#4a3aa7",  # slot 5 violet
    "GRPO": "#e34948",  # slot 6 red
    "Teacher": "#eb6834",  # slot 8 orange
}


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE + 3,
            "axes.labelsize": BASE_FONT_SIZE + 1,
            "xtick.labelsize": BASE_FONT_SIZE,
            "ytick.labelsize": BASE_FONT_SIZE,
            "legend.fontsize": BASE_FONT_SIZE - 2,
            "figure.titlesize": BASE_FONT_SIZE + 3,
        }
    )


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def smooth(values: list[float], weight: float) -> list[float]:
    result, last = [], None
    for v in values:
        last = v if last is None else last * weight + v * (1 - weight)
        result.append(last)
    return result


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


LEVELS = list(range(1, 9))


def plot_accuracy_by_level(rows: dict[str, list[float]], colors: dict[str, str], title: str, out_name: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, accs in rows.items():
        ax.plot(LEVELS, accs, marker="o", linewidth=2, markersize=6, label=label, color=colors[label])
    ax.set_xlabel("Composition Level")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(LEVELS)
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, out_name)


def mimic_zone_figure():
    # Base, Bootstrap-SFT, Teacher-SFT, On-policy-SFT accuracy by level (results/main.csv, results/baseline.csv)
    data = {
        "Base": [0.74609375, 0.125, 0.0234375, 0.0078125, 0.00390625, 0, 0, 0],
        "Bootstrap": [0.7421875, 0.2421875, 0.05859375, 0.02734375, 0.01171875, 0, 0, 0],
        "Teacher": [0.77734375, 0.3671875, 0.08984375, 0.03125, 0, 0, 0, 0],
        "On-policy": [0.69140625, 0.28515625, 0.06640625, 0.02734375, 0.01171875, 0, 0, 0],
    }
    plot_accuracy_by_level(data, COLOR, "SFT accuracy by data source", "data_vary_loss_fn_sft_no_onpolicy.png")


def generalization_zone_figure():
    data = {
        "SFT": [0.69140625, 0.28515625, 0.06640625, 0.02734375, 0.01171875, 0, 0, 0],
        "POS+NEG": [0.78125, 0.54296875, 0.1328125, 0.046875, 0.0078125, 0.00390625, 0, 0],
        "REINFORCE+Baseline": [0.875, 0.67188, 0.27734, 0.078125, 0.015625, 0.0078125, 0.0039063, 0],
        "GRPO": [0.88281, 0.73047, 0.25781, 0.11719, 0.023438, 0.011719, 0.0039, 0],
    }
    plot_accuracy_by_level(data, COLOR, "Accuracy by loss function, on-policy data", "loss_fn_on_policy.png")


def response_length_figure():
    methods = ["SFT", "POS+NEG", "REINFORCE+Baseline", "GRPO"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for method in methods:
        path = RESULTS_DIR / f"string_task_onpolicy_{method.replace('+', '')}_dense.csv"
        rows = read_csv(path)
        steps = [int(r["_step"]) for r in rows if r["reward/score/mean"]]
        acc = [float(r["reward/score/mean"]) for r in rows if r["reward/score/mean"]]
        len_steps = [int(r["_step"]) for r in rows if r["rollout/avg_response_length"]]
        lens = [float(r["rollout/avg_response_length"]) for r in rows if r["rollout/avg_response_length"]]
        ax1.plot(steps, smooth(acc, 0.95), linewidth=1.8, label=method, color=COLOR[method])
        ax2.plot(len_steps, smooth(lens, 0.95), linewidth=1.8, label=method, color=COLOR[method])
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Train-batch accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Avg response length (tokens)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    save(fig, "response_length.png")


def entropy_figure():
    methods = ["SFT", "POS+NEG", "REINFORCE+Baseline", "GRPO"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        path = RESULTS_DIR / f"string_task_onpolicy_{method.replace('+', '')}_entropy.csv"
        rows = read_csv(path)
        key = "val/16-codeio-forward-incomplete-depth2/entropy/avg"
        steps = [int(r["_step"]) for r in rows if r[key]]
        vals = [float(r[key]) for r in rows if r[key]]
        ax.plot(steps, vals, marker="o", markersize=4, linewidth=1.8, label=method, color=COLOR[method])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Eval entropy (depth-2, avg)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "entropy.png")


def grad_norm_figure():
    methods = ["SFT", "POS+NEG", "REINFORCE+Baseline", "GRPO"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        path = RESULTS_DIR / f"string_task_onpolicy_{method.replace('+', '')}_dense.csv"
        rows = read_csv(path)
        steps = [int(r["_step"]) for r in rows if r["actor/grad_norm"]]
        vals = [float(r["actor/grad_norm"]) for r in rows if r["actor/grad_norm"]]
        ax.plot(steps, vals, linewidth=1.2, alpha=0.8, label=method, color=COLOR[method])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient norm")
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "grad_norm.png")


def grad_norm_collapse_figure():
    onpolicy_methods = ["SFT", "POS+NEG", "REINFORCE+Baseline", "GRPO"]
    offpolicy_runs = {
        "Bootstrap-GRPO": ("string_task_offpolicy_Bootstrap_GRPO_gradnorm.csv", COLOR["Bootstrap"], "-"),
        "Bootstrap-REINFORCE+Baseline": ("string_task_offpolicy_Bootstrap_REINFORCEBaseline_gradnorm.csv", COLOR["Bootstrap"], "--"),
        "Teacher-GRPO": ("string_task_offpolicy_Teacher_GRPO_gradnorm.csv", COLOR["Teacher"], "-"),
        "Teacher-REINFORCE+Baseline": ("string_task_offpolicy_Teacher_REINFORCEBaseline_gradnorm.csv", COLOR["Teacher"], "--"),
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in onpolicy_methods:
        path = RESULTS_DIR / f"string_task_onpolicy_{method.replace('+', '')}_dense.csv"
        rows = read_csv(path)
        steps = [int(r["_step"]) for r in rows if r["actor/grad_norm"]]
        vals = [float(r["actor/grad_norm"]) for r in rows if r["actor/grad_norm"]]
        ax.plot(steps, vals, linewidth=1.0, alpha=0.6, color=COLOR["On-policy"])
    ax.plot([], [], linewidth=1.5, color=COLOR["On-policy"], label="On-policy (all losses)")

    for label, (fname, color, style) in offpolicy_runs.items():
        rows = read_csv(RESULTS_DIR / fname)
        steps = [int(r["_step"]) for r in rows if r["actor/grad_norm"]]
        vals = [float(r["actor/grad_norm"]) for r in rows if r["actor/grad_norm"]]
        ax.plot(steps, vals, linewidth=1.8, linestyle=style, color=color, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient norm (log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=BASE_FONT_SIZE - 4)
    fig.tight_layout()
    save(fig, "grad_norm_collapse.png")


def math_bootstrap_sft_figure():
    rows = read_csv(RESULTS_DIR / "math_bootstrap_sft_easy_pass1.csv")
    keys = {
        "easy": "val-core/math-easy/reward/pass@1",
        "medium": "val-core/math-medium/reward/pass@1",
        "hard": "val-core/math-hard/reward/pass@1",
    }
    colors = {"easy": COLOR["GRPO"], "medium": COLOR["POS+NEG"], "hard": COLOR["REINFORCE+Baseline"]}
    fig, ax = plt.subplots(figsize=(8, 5))
    for split, key in keys.items():
        steps = [int(r["_step"]) for r in rows if r[key]]
        vals = [float(r[key]) for r in rows if r[key]]
        ax.plot(steps, vals, marker="o", markersize=5, linewidth=1.8, label=split, color=colors[split])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Pass@1")
    ax.set_title("Math Bootstrap-SFT (trained on easy only)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "math_bootstrap_sft_pass1.png")


def main():
    configure_plot_style()
    mimic_zone_figure()
    generalization_zone_figure()
    response_length_figure()
    entropy_figure()
    grad_norm_figure()
    grad_norm_collapse_figure()
    math_bootstrap_sft_figure()


if __name__ == "__main__":
    main()
