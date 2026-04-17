import sys
import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from verl.utils.reward_score import _default_compute_score
from tqdm import tqdm

# Mapping from func_N identifiers to human-readable names (from string_data.py)
FUNC_NAME_MAPPING = {
    "func_0": "deterministic_shuffle",
    "func_1": "repeat_str",
    "func_2": "remove_vowels",
    "func_3": "sort_chars",
    "func_4": "reverse_words",
    "func_5": "add_prefix",
    "func_6": "add_suffix",
    "func_7": "interlace_str",
    "func_8": "rotate_str",
    "func_9": "mirror_str",
    "func_10": "alternate_case",
    "func_11": "shift_chars",
    "func_12": "vowel_to_number",
    "func_13": "insert_separator",
    "func_14": "duplicate_every_char",
    "func_15": "fancy_brackets",
    "func_16": "compress_repeats",
    "func_17": "recursive_reverse",
    "func_18": "loop_concat",
    "func_19": "while_rotate",
    "func_20": "recursive_interlace",
    "func_21": "loop_filter_nonalpha",
    "func_22": "verify_even_length",
    "func_23": "backchain_add_digit",
    "func_24": "backchain_palindrome",
}

FUNC_PATTERN = re.compile(r'\b(func_\d+)\b')

def compute_accuracy(dataset):
    prompts = dataset["prompt"]
    responses = dataset["responses"]
    data_sources = dataset["data_source"]
    reward_model_data = dataset["reward_model"]

    total = len(dataset)
    total_scores = []
    passes = 0

    # Per data_source: accumulate score_lists, passes, and total count
    by_source = {}  # data_source -> {"score_lists": [], "passes": 0, "total": 0}

    for i in tqdm(range(total)):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        ground_truth = reward_data["ground_truth"]
        score_lst = []
        for r in response_lst:
            score = _default_compute_score(data_source, r, ground_truth)[1]
            score_lst.append(score)
        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1

        if data_source not in by_source:
            by_source[data_source] = {"score_lists": [], "passes": 0, "total": 0}
        by_source[data_source]["score_lists"].append(score_lst)
        by_source[data_source]["total"] += 1
        if max_score == 1:
            by_source[data_source]["passes"] += 1

    pass_at_n = passes / total if total else 0.0
    pass_at_1 = float(np.mean(total_scores)) if total_scores else 0.0

    by_data_source = {}
    for source, d in by_source.items():
        t = d["total"]
        plists = d["score_lists"]
        by_data_source[source] = {
            "pass_at_1": float(np.mean(plists)) if plists else 0.0,
            "pass_at_n": d["passes"] / t if t else 0.0,
        }

    return {
        "pass_at_1": pass_at_1,
        "pass_at_n": pass_at_n,
        "by_data_source": by_data_source,
    }

def extract_functions_from_prompt(prompt) -> tuple:
    """
    Parse the prompt (list of message dicts or a string) to extract the ordered
    tuple of func_N identifiers used in the main_solution return expression.

    For depth=1 the tuple has one element, e.g. ('func_3',).
    For depth>1 the tuple lists functions in the order they appear in the return
    expression (outermost first), e.g. ('func_3', 'func_9').
    """
    if isinstance(prompt, (list, np.ndarray)):
        content = " ".join(
            msg["content"] if isinstance(msg, dict) else str(msg)
            for msg in prompt
        )
    else:
        content = str(prompt)

    # Find the main_solution definition and grab its return line
    match = re.search(r'def main_solution\([^)]*\):\s*return\s+(.+)', content)
    if match:
        return_expr = match.group(1).strip()
        funcs = FUNC_PATTERN.findall(return_expr)
        if funcs:
            return tuple(funcs)

    # Fallback: scan the entire code block for func_N references (excludes definitions)
    funcs = []
    for m in FUNC_PATTERN.finditer(content):
        start = m.start()
        preceding = content[max(0, start - 4):start]
        if preceding != 'def ':
            funcs.append(m.group(1))
    if funcs:
        return tuple(dict.fromkeys(funcs))  # deduplicated, insertion-ordered

    return ()


def analyze_by_function(dataset):
    """
    Compute pass@1 and pass@n accuracy grouped by the string function(s) used
    in each problem.

    For depth=1 problems the group key is the single function name (str).
    For depth>1 problems the group key is a tuple of function names in the order
    they appear in the return expression (e.g. ('func_3', 'func_9')).

    Returns a dict mapping each group key to {"pass_at_1": float, "pass_at_n": float, "total": int}.
    Human-readable function names are also included via FUNC_NAME_MAPPING.
    """
    prompts = dataset["prompt"]
    responses = dataset["responses"]
    data_sources = dataset["data_source"]
    reward_model_data = dataset["reward_model"]

    total = len(dataset)

    # group key -> {"score_lists": [...], "passes": int, "total": int}
    by_func: dict = defaultdict(lambda: {"score_lists": [], "passes": 0, "total": 0})

    for i in tqdm(range(total), desc="analyze_by_function"):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        ground_truth = reward_data["ground_truth"]

        score_lst = []
        for r in response_lst:
            score = _default_compute_score(data_source, r, ground_truth)[1]
            score_lst.append(score)
        max_score = np.max(score_lst)

        func_key = extract_functions_from_prompt(prompt)

        by_func[func_key]["score_lists"].append(score_lst)
        by_func[func_key]["total"] += 1
        if max_score == 1:
            by_func[func_key]["passes"] += 1

    result = {}
    for func_key, d in sorted(by_func.items(), key=lambda x: (len(x[0]), x[0])):
        t = d["total"]
        plists = d["score_lists"]
        if len(func_key) == 1:
            readable_key = FUNC_NAME_MAPPING.get(func_key[0], func_key[0])
        else:
            readable_key = tuple(FUNC_NAME_MAPPING.get(f, f) for f in func_key)

        result[str(readable_key)] = {
            "func_ids": list(func_key),
            "pass_at_1": float(np.mean(plists)) if plists else 0.0,
            "pass_at_n": d["passes"] / t if t else 0.0,
            "total": t,
        }

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <parquet_file> <output_file>")
        sys.exit(1)

    parquet_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load parquet file
    df = pd.read_parquet(parquet_file)

    accuracy = compute_accuracy(df)
    print(accuracy)

    by_function = analyze_by_function(df)
    accuracy["by_function"] = by_function

    # Save accuracy results to a JSON file
    with open(output_file, "w") as f:
        json.dump(accuracy, f, indent=2)
    print(f"Accuracy results saved to {output_file}")


if __name__ == "__main__":
    main()