#!/usr/bin/env python3
import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit an sbatch job using a template file. Replaces #NAME and #SCRIPT."
        ),
        epilog=(
            "Extra sbatch flags (e.g. --exclude=babel-p5-24, --nodelist=babel-x9-16) "
            "can be appended after the script path and are passed through to `sbatch` "
            "unchanged. Put --template before the script path, since anything after "
            "the script path is treated as a raw sbatch flag, e.g.:\n"
            "  python deploy/launcher.py --template t.sbatch script.sh --exclude=babel-p5-24"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "script_path",
        type=Path,
        help="Path to the bash script to run (e.g. experiments/exp1/exp1-1.sh).",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("template.sbatch"),
        help="Path to sbatch template (default: deploy/template.sbatch).",
    )
    parser.add_argument(
        "sbatch_args",
        nargs=argparse.REMAINDER,
        help=(
            "Extra arguments passed through to `sbatch` unchanged, e.g. "
            "--exclude=babel-p5-24 or --nodelist=babel-x9-16."
        ),
    )
    return parser.parse_args()


def render_template(template_text: str, job_name: str, script_path: Path) -> str:
    return (
        template_text.replace("#NAME", job_name).replace("#SCRIPT", str(script_path))
    )


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.template):
        raise FileNotFoundError(f"Template not found: {args.template}")
    if not os.path.exists(args.script_path):
        raise FileNotFoundError(f"Script not found: {args.script_path}")
    template_path = args.template
    script_path = args.script_path
    job_name = script_path.stem

    template_text = template_path.read_text(encoding="utf-8")
    sbatch_contents = render_template(template_text, job_name, script_path)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".sbatch",
        prefix="launcher-",
        delete=True,
        encoding="utf-8",
    ) as temp_sbatch:
        temp_sbatch.write(sbatch_contents)
        temp_sbatch.flush()
        result = subprocess.run(
            ["sbatch", *args.sbatch_args, temp_sbatch.name],
            check=False,
            text=True,
            capture_output=True,
        )

    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.strip())
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
