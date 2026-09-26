"""Fetch training-dynamics metrics (grad norm, perplexity, entropy, response
length) directly from the public wandb projects (no API key needed for public
project GraphQL reads) and cache them as CSVs in results/.

String-task runs on the cluster get preempted and resumed under fresh wandb
run IDs, so each on-policy method here is a *chain* of 2-3 run segments
stitched together by step (later segment wins on overlap). The chains below
were identified by walking the string-task project's run list and picking, for
each (data-source x loss) cell, the runs whose step ranges tile a contiguous
0->final trajectory.

Usage:
    WANDB_ENTITY=<entity> python scripts/analysis/fetch_wandb_training_dynamics.py
"""

import csv
import json
import os
import urllib.request
from pathlib import Path

API_URL = "https://api.wandb.ai/graphql"
RESULTS_DIR = Path("results")

RUN_QUERY = """
query Run($entityName: String!, $projectName: String!, $runName: String!, $specs: [JSONString!]!) {
  project(entityName: $entityName, name: $projectName) {
    run(name: $runName) { sampledHistory(specs: $specs) }
  }
}
"""

# On-policy string-task runs are resumed chains; earlier id first.
STRING_TASK_ONPOLICY_CHAINS = {
    "SFT": ["8dtgngsw", "0ujssfhy", "tn8vt0on"],
    "POS+NEG": ["8wcbj9qs", "g6tcep2o", "9i53r4nn"],
    "REINFORCE+Baseline": ["xrvovrpg", "4a0lj4bf"],
    "GRPO": ["w62sniyp", "dvh23ze5"],
}

# Off-policy (frozen-pool) runs that collapsed -- single run each, no resume.
STRING_TASK_OFFPOLICY_COLLAPSE_RUNS = {
    "Bootstrap-GRPO": "azc43oji",
    "Bootstrap-REINFORCE+Baseline": "tbq3u1no",
    "Teacher-GRPO": "6hkzig3d",
    "Teacher-REINFORCE+Baseline": "fd66oamx",
}

MATH_BOOTSTRAP_SFT_RUN = "297gy4rz"

ENTROPY_KEY = "val/16-codeio-forward-incomplete-depth2/entropy/avg"

# On-policy math-task runs are also resumed chains (same cluster-preemption pattern
# as string-task); earlier id first.
MATH_TASK_ONPOLICY_CHAINS = {
    "GRPO": ["7yqjetyk", "2zpizahh"],
    "PosNeg": ["z3qmq1df", "en0qvey1", "cfhvlll4"],
    "SFT": ["xjx8xkjg", "fnpd7ype"],
    "ReinforceBaseline": ["ztfye15q"],
}

MATH_ENTROPY_KEYS = [
    "_step",
    "val/16-math-easy/entropy/avg",
    "val/16-math-medium/entropy/avg",
    "val/16-math-hard/entropy/avg",
]


def gql(query: str, variables: dict) -> dict:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def fetch_history(entity: str, project: str, run_id: str, keys: list[str], samples: int = 3000) -> list[dict]:
    spec = json.dumps({"keys": keys, "samples": samples})
    data = gql(RUN_QUERY, {"entityName": entity, "projectName": project, "runName": run_id, "specs": [spec]})
    run = data["data"]["project"]["run"]
    if run is None:
        return []
    return run["sampledHistory"][0]


def fetch_chain(entity: str, project: str, chain: list[str], keys: list[str]) -> list[dict]:
    """Stitch a resumed-run chain together by step; later runs in the chain win on overlap."""
    by_step: dict[int, dict] = {}
    for run_id in chain:
        for row in fetch_history(entity, project, run_id, keys):
            by_step[row["_step"]] = row
    return sorted(by_step.values(), key=lambda r: r["_step"])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {path} ({len(rows)} rows)")


def main():
    entity = os.environ["WANDB_ENTITY"]

    # 1. On-policy string-task training dynamics: dense (grad_norm/perplexity/response_length/score)
    dense_keys = ["_step", "actor/grad_norm", "actor/perplexity", "rollout/avg_response_length", "reward/score/mean"]
    for method, chain in STRING_TASK_ONPOLICY_CHAINS.items():
        rows = fetch_chain(entity, "string-task", chain, dense_keys)
        write_csv(RESULTS_DIR / f"string_task_onpolicy_{method.replace('+', '')}_dense.csv", rows, dense_keys)

    # 2. On-policy string-task entropy (sparser -- only logged at eval steps)
    entropy_keys = ["_step", ENTROPY_KEY]
    for method, chain in STRING_TASK_ONPOLICY_CHAINS.items():
        rows = fetch_chain(entity, "string-task", chain, entropy_keys)
        write_csv(RESULTS_DIR / f"string_task_onpolicy_{method.replace('+', '')}_entropy.csv", rows, entropy_keys)

    # 3. Off-policy collapse runs: grad_norm/perplexity (single run each, no resume)
    offpolicy_keys = ["_step", "actor/grad_norm", "actor/perplexity"]
    for label, run_id in STRING_TASK_OFFPOLICY_COLLAPSE_RUNS.items():
        rows = fetch_history(entity, "string-task", run_id, offpolicy_keys, samples=3000)
        write_csv(RESULTS_DIR / f"string_task_offpolicy_{label.replace('+', '').replace('-', '_')}_gradnorm.csv", rows, offpolicy_keys)

    # 4. Math-task Bootstrap-SFT-easy eval curve (easy/medium/hard pass@1)
    math_keys = [
        "_step",
        "val-core/math-easy/reward/pass@1",
        "val-core/math-medium/reward/pass@1",
        "val-core/math-hard/reward/pass@1",
    ]
    rows = fetch_history(entity, "math-task", MATH_BOOTSTRAP_SFT_RUN, math_keys, samples=3000)
    write_csv(RESULTS_DIR / "math_bootstrap_sft_easy_pass1.csv", rows, math_keys)

    # 5. On-policy math-task entropy (easy/medium/hard splits)
    for method, chain in MATH_TASK_ONPOLICY_CHAINS.items():
        rows = fetch_chain(entity, "math-task", chain, MATH_ENTROPY_KEYS)
        write_csv(RESULTS_DIR / f"math_onpolicy_{method}_entropy.csv", rows, MATH_ENTROPY_KEYS)


if __name__ == "__main__":
    main()
