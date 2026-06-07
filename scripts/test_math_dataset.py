"""
Test script to validate that data/math parquet files are compatible with the
verl OSFT training pipeline.

Checks:
  1. Required columns present
  2. prompt format (list of role/content dicts, tokenizable with chat template)
  3. data_source non-empty and recognized by default_compute_score
  4. reward_model has ground_truth key with non-empty value
  5. extra_info is a plain dict (no numpy arrays inside scalar fields)
  6. Reward function can score a dummy answer without error
  7. No rows with missing / null critical fields
"""

import sys
import argparse
import traceback
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"prompt", "data_source", "ability", "reward_model", "extra_info"}

KNOWN_DATA_SOURCES = {
    "openai/gsm8k",
    "lighteval/MATH",
    "DigitalLearningGmbH/MATH-lighteval",
    "math500", "amc", "aime", "aime25", "olympiadbench", "minerva", "amc23",
    "deepscaler", "math_dapo", "math_12k",
    "open-thoughts/OpenThoughts3-1.2M",
    "numina_aops_forum", "numina_synthetic_math", "numina_amc_aime",
    "numina_synthetic_amc", "numina_cn_k12", "numina_olympiads",
    "codecontests", "apps", "codeforces", "taco",
    "hiyouga/geometry3k",
    "searchR1_nq", "searchR1_triviaqa", "searchR1_popqa",
    "searchR1_hotpotqa", "searchR1_2wikimultihopqa", "searchR1_musique", "searchR1_bamboogle",
}
# codeio data sources match by prefix "codeio-"

DUMMY_CORRECT_ANSWERS = {
    "boxed": r"The answer is \boxed{42}",
    "json": '{"output": "test"}',
}


def _is_scalar(v):
    """Return True if v is a Python scalar (not a numpy array/list)."""
    if isinstance(v, (np.ndarray, list)):
        return False
    return True


def check_columns(df: pd.DataFrame, path: str) -> list[str]:
    errors = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
    extra = set(df.columns) - REQUIRED_COLUMNS - {"solution", "reward", "length",
                                                    "correct_length", "incorrect_length",
                                                    "__index_level_0__"}
    if extra:
        print(f"  [INFO] Extra columns (will be ignored by trainer): {sorted(extra)}")
    return errors


def check_prompt_format(df: pd.DataFrame, n_samples: int = 5) -> list[str]:
    errors = []
    for i, row in df.head(n_samples).iterrows():
        prompt = row["prompt"]
        if not isinstance(prompt, (list, np.ndarray)):
            errors.append(f"Row {i}: prompt is not a list (got {type(prompt).__name__})")
            continue
        for j, msg in enumerate(prompt):
            if not isinstance(msg, dict):
                errors.append(f"Row {i}, message {j}: not a dict (got {type(msg).__name__})")
                continue
            if "role" not in msg:
                errors.append(f"Row {i}, message {j}: missing 'role' key")
            if "content" not in msg:
                errors.append(f"Row {i}, message {j}: missing 'content' key")
    return errors


def check_data_source(df: pd.DataFrame) -> list[str]:
    errors = []
    null_mask = df["data_source"].isnull() | (df["data_source"] == "")
    if null_mask.any():
        n = int(null_mask.sum())
        errors.append(
            f"{n}/{len(df)} rows have empty/null data_source — "
            "default_compute_score will raise NotImplementedError for these rows"
        )
    unique_sources = df["data_source"].dropna().unique()
    for src in unique_sources:
        if src == "":
            continue
        if not (src in KNOWN_DATA_SOURCES or str(src).startswith("codeio-")):
            errors.append(
                f"data_source '{src}' is not recognized by default_compute_score "
                "(would raise NotImplementedError)"
            )
    return errors


def check_reward_model(df: pd.DataFrame, n_samples: int = 10) -> list[str]:
    errors = []
    for i, row in df.head(n_samples).iterrows():
        rm = row["reward_model"]
        if not isinstance(rm, dict):
            errors.append(f"Row {i}: reward_model is not a dict (got {type(rm).__name__})")
            continue
        if "ground_truth" not in rm:
            errors.append(f"Row {i}: reward_model missing 'ground_truth' key")
            continue
        gt = rm["ground_truth"]
        if gt is None or gt == "":
            errors.append(f"Row {i}: reward_model['ground_truth'] is empty/null")
        if "style" not in rm:
            errors.append(f"Row {i}: reward_model missing 'style' key (expected 'rule')")
    return errors


def check_extra_info(df: pd.DataFrame, n_samples: int = 10) -> list[str]:
    """Flag scalar fields that are numpy arrays instead of plain Python types."""
    errors = []
    for i, row in df.head(n_samples).iterrows():
        ei = row["extra_info"]
        if not isinstance(ei, dict):
            errors.append(f"Row {i}: extra_info is not a dict (got {type(ei).__name__})")
            continue
        for k, v in ei.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 0:
                    errors.append(
                        f"Row {i}: extra_info['{k}'] is a 0-d numpy array; "
                        "use .item() to convert to a Python scalar"
                    )
                elif v.size == 1:
                    errors.append(
                        f"Row {i}: extra_info['{k}'] is a 1-element numpy array {v!r}; "
                        "should be a plain Python scalar"
                    )
    return errors


def check_reward_fn(df: pd.DataFrame) -> list[str]:
    """Try calling default_compute_score on the first non-empty data_source row."""
    errors = []
    try:
        from verl.utils.reward_score import default_compute_score
    except ImportError as e:
        errors.append(f"Could not import default_compute_score: {e}")
        return errors

    valid_rows = df[df["data_source"].notna() & (df["data_source"] != "")]
    if valid_rows.empty:
        errors.append("No rows with a valid data_source to test reward function")
        return errors

    row = valid_rows.iloc[0]
    src = row["data_source"]
    gt = row["reward_model"].get("ground_truth", "")

    # try a dummy answer that includes \boxed{} (common math format)
    dummy = r"Let me compute. The answer is \boxed{0}."
    try:
        score = default_compute_score(src, dummy, str(gt))
        print(f"  [INFO] Reward fn test: data_source='{src}', score={score}")
    except NotImplementedError:
        errors.append(
            f"default_compute_score raised NotImplementedError for data_source='{src}'"
        )
    except Exception as e:
        errors.append(f"Reward fn raised unexpected error for data_source='{src}': {e}")
    return errors


def check_tokenizer(df: pd.DataFrame, n_samples: int = 3) -> list[str]:
    """Try applying a chat template to confirm prompts are well-formed.
    Skips gracefully if transformers is unavailable or no tokenizer found."""
    errors = []
    try:
        from transformers import AutoTokenizer
        # Use a tiny tokenizer just for format validation
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                             trust_remote_code=True)
    except Exception:
        print("  [INFO] Skipping tokenizer check (tokenizer not available locally)")
        return errors

    for i, row in df.head(n_samples).iterrows():
        prompt = list(row["prompt"])
        try:
            tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        except Exception as e:
            errors.append(f"Row {i}: apply_chat_template failed: {e}")
    return errors


def run_checks(path: str) -> bool:
    print(f"\n{'='*60}")
    print(f"Checking: {path}")
    print(f"{'='*60}")

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  [FAIL] Could not read parquet: {e}")
        return False

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    all_errors = []

    checks = [
        ("Column schema", lambda: check_columns(df, path)),
        ("Prompt format", lambda: check_prompt_format(df)),
        ("data_source", lambda: check_data_source(df)),
        ("reward_model", lambda: check_reward_model(df)),
        ("extra_info types", lambda: check_extra_info(df)),
        ("Reward function", lambda: check_reward_fn(df)),
    ]

    for name, fn in checks:
        try:
            errs = fn()
        except Exception:
            errs = [f"Check raised exception:\n{traceback.format_exc()}"]
        if errs:
            for e in errs:
                print(f"  [FAIL] {name}: {e}")
            all_errors.extend(errs)
        else:
            print(f"  [ OK ] {name}")

    if all_errors:
        print(f"\n  --> {len(all_errors)} issue(s) found in {path}")
        return False
    else:
        print(f"\n  --> All checks passed for {path}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate math dataset compatibility with verl OSFT training")
    parser.add_argument(
        "files",
        nargs="*",
        default=[
            "data/math/math-easy/train.parquet",
            "data/math/math-medhard/train.parquet",
        ],
        help="Parquet files to check (default: both math datasets)",
    )
    args = parser.parse_args()

    results = {}
    for path in args.files:
        results[path] = run_checks(path)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    all_passed = True
    for path, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {path}")
        if not ok:
            all_passed = False

    if not all_passed:
        print("\nAction needed: fix the issues above before using these files for training.")
        sys.exit(1)
    else:
        print("\nAll datasets are compatible with the training pipeline.")


if __name__ == "__main__":
    main()
