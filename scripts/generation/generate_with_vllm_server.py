#!/usr/bin/env python3
"""
Load prompts from parquet, call a vLLM OpenAI endpoint, and save generations.
"""

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses using a running vLLM server.")
    parser.add_argument("--data-path", required=True, help="Input parquet path.")
    parser.add_argument("--output-path", required=True, help="Output parquet path with generations.")
    parser.add_argument("--prompt-key", default="prompt", help="Column name for input prompts.")
    parser.add_argument("--response-key", default="responses", help="Column name to write generated responses.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000", help="vLLM OpenAI server base URL.")
    parser.add_argument("--model", default=None, help="Model name in server. Auto-detected if omitted.")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of samples per prompt.")
    parser.add_argument("--batch-n", type=int, default=None, help="Max samples per HTTP request. n-samples is split into ceil(n-samples/batch-n) calls and results are merged. Defaults to n-samples (single request).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generated tokens per sample.")
    parser.add_argument("--num-workers", type=int, default=8, help="Concurrent request workers.")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout per request in seconds.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retry attempts per sample request.")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Exponential backoff base.")
    parser.add_argument(
        "--stop",
        default=None,
        help='Stop string or JSON list string, e.g. "</answer>" or \'["</answer>", "<|eot_id|>"]\'.',
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to JSONL checkpoint file. Defaults to <output-path>.checkpoint.jsonl",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and start from scratch.",
    )
    return parser.parse_args()


def _http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    if payload is None:
        request = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
        body = None
    else:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_model_name(server_url: str, user_model: str | None, timeout: float) -> str:
    if user_model:
        return user_model
    models_payload = _http_json(f"{server_url.rstrip('/')}/v1/models", payload=None, timeout=timeout)
    models = models_payload.get("data", [])
    if not models:
        raise RuntimeError(
            "Could not auto-detect model from /v1/models; pass --model explicitly."
        )
    return models[0]["id"]


def normalize_messages(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
        return prompt
    if isinstance(prompt, dict) and isinstance(prompt.get("messages"), list):
        return prompt["messages"]
    return [{"role": "user", "content": str(prompt)}]


def parse_stop(stop_raw: str | None) -> str | list[str] | None:
    if stop_raw is None:
        return None
    stop_raw = stop_raw.strip()
    if stop_raw.startswith("["):
        parsed = json.loads(stop_raw)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("--stop JSON must decode to list[str].")
        return parsed
    return stop_raw


def _request_n(
    url: str,
    messages: list[dict],
    n: int,
    model: str,
    args: argparse.Namespace,
    stop: str | list[str] | None,
) -> list[str]:
    payload = {
        "model": model,
        "messages": messages,
        "n": n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if stop is not None:
        payload["stop"] = stop

    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            result = _http_json(url, payload=payload, timeout=args.timeout)
            choices = result.get("choices", [])
            texts = [choice["message"]["content"] for choice in choices]
            if len(texts) != n:
                raise RuntimeError(f"Expected {n} samples, got {len(texts)}.")
            return texts
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == args.max_retries:
                break
            sleep_seconds = args.retry_backoff ** (attempt - 1)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed after {args.max_retries} attempts: {last_error}") from last_error


def generate_one(
    prompt: Any,
    model: str,
    args: argparse.Namespace,
    stop: str | list[str] | None,
) -> list[str]:
    url = f"{args.server_url.rstrip('/')}/v1/chat/completions"
    messages = normalize_messages(prompt)
    batch_n = args.batch_n if args.batch_n is not None else args.n_samples

    results: list[str] = []
    remaining = args.n_samples
    while remaining > 0:
        n = min(batch_n, remaining)
        results.extend(_request_n(url, messages, n, model, args, stop))
        remaining -= n
    return results


class CheckpointManager:
    """Thread-safe append-only JSONL checkpoint. Each record is one line."""

    def __init__(self, path: str, total: int) -> None:
        self.path = path
        self.total = total
        self._lock = threading.Lock()
        self._responses: dict[int, list[str]] = {}
        self._errors: list[tuple[int, str]] = []
        self._fh: object = None

    def load(self) -> dict[int, list[str]]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "responses" in entry:
                    self._responses[entry["idx"]] = entry["responses"]
                else:
                    self._errors.append((entry["idx"], entry["error"]))
        print(f"Resumed from checkpoint: {len(self._responses)}/{self.total} already done ({self.path})")
        return dict(self._responses)

    def record(self, idx: int, result: list[str] | None, error: str | None) -> None:
        with self._lock:
            if result is not None:
                self._responses[idx] = result
                entry = {"idx": idx, "responses": result}
            else:
                self._errors.append((idx, error))
                entry = {"idx": idx, "error": error}
            if self._fh is None:
                self._fh = open(self.path, "a")
            self._fh.write(json.dumps(entry) + "\n")
            self._fh.flush()

    def save(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def responses(self) -> dict[int, list[str]]:
        with self._lock:
            return dict(self._responses)

    def errors(self) -> list[tuple[int, str]]:
        with self._lock:
            return list(self._errors)

    def delete(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
        if os.path.exists(self.path):
            os.remove(self.path)


def main() -> None:
    args = parse_args()
    if os.path.exists(args.output_path) and not args.overwrite:
        raise FileExistsError(f"{args.output_path} already exists. Use --overwrite to replace.")
    if args.n_samples < 1:
        raise ValueError("--n-samples must be >= 1.")
    if args.temperature == 0.0 and args.n_samples > 1:
        raise ValueError("For deterministic decoding (--temperature 0), set --n-samples=1.")

    checkpoint_path = args.checkpoint_path or (args.output_path + ".checkpoint.jsonl")

    stop = parse_stop(args.stop)
    model = resolve_model_name(args.server_url, args.model, args.timeout)
    print(f"Using model: {model}")

    dataset = load_dataset("parquet", data_files=args.data_path)["train"]
    prompts = dataset[args.prompt_key]
    total = len(prompts)
    print(f"Loaded {total} prompts from {args.data_path}")

    ckpt = CheckpointManager(checkpoint_path, total)

    already_done: dict[int, list[str]] = {}
    if not args.no_resume:
        already_done = ckpt.load()

    pending_indices = [i for i in range(total) if i not in already_done]
    print(f"Generating {len(pending_indices)} remaining prompts "
          f"({len(already_done)} already completed)")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_index = {
            executor.submit(generate_one, prompt=prompts[idx], model=model, args=args, stop=stop): idx
            for idx in pending_indices
        }
        for future in tqdm(as_completed(future_to_index), total=len(pending_indices), desc="Generating"):
            idx = future_to_index[future]
            try:
                ckpt.record(idx, future.result(), None)
            except Exception as exc:  # noqa: BLE001
                ckpt.record(idx, None, str(exc))

    ckpt.save()

    errors = ckpt.errors()
    if errors:
        sample_errors = "\n".join(f"idx={idx}: {msg}" for idx, msg in errors[:20])
        raise RuntimeError(
            f"Generation failed for {len(errors)} / {total} samples.\n"
            f"Checkpoint saved to {checkpoint_path} — re-run to resume.\n"
            f"First errors:\n{sample_errors}"
        )

    all_responses = ckpt.responses()
    if len(all_responses) != total:
        raise RuntimeError(
            f"Internal error: have {len(all_responses)} responses for {total} prompts."
        )

    final_responses = [all_responses[i] for i in range(total)]

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dataset = dataset.add_column(args.response_key, final_responses)
    dataset.to_parquet(args.output_path)
    print(f"Saved generations to {args.output_path}")

    ckpt.delete()
    print(f"Deleted checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()