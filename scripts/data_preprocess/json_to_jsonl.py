import json
import sys
from pathlib import Path

def json_to_jsonl(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON structure must be a list.")

    with output_path.open("w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Wrote {len(data)} records to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_jsonl.py input.json output.jsonl")
        sys.exit(1)

    json_to_jsonl(sys.argv[1], sys.argv[2])