"""Upload the code to Kaggle and run the pipeline there.

    python scripts/kaggle_push.py code      # create / new version of dataset <user>/gar-code
    python scripts/kaggle_push.py kernel    # push + run kaggle/run.py on a T4 (kernel <user>/gar-pipeline)
    python scripts/kaggle_push.py kernel --name gar-ablation --reuse gar-pipeline --steps b8_avg transformer
                                            # another kernel, starting from gar-pipeline's features/checkpoints
    python scripts/kaggle_push.py status
    python scripts/kaggle_push.py output    # download the kernel output into results_kaggle/<kernel>/

Uses the Kaggle CLI 2.x from D:/tools/kaggle-cli. Token: ~/.kaggle/access_token, or KAGGLE_API_TOKEN;
for another account also set GAR_KAGGLE_USER to its username.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

KAGGLE = "D:/tools/kaggle-cli/Scripts/kaggle.exe"
USER = os.environ.get("GAR_KAGGLE_USER", "idrissjedid")  # owner of the code dataset + kernel
CODE_DATASET = f"{USER}/gar-code"
KERNEL = f"{USER}/gar-pipeline"
SOURCES = ["sherif31/group-activity-recognition-volleyball", CODE_DATASET]

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "kaggle"


def kaggle(*args, check=True):
    print("$ kaggle " + " ".join(args), flush=True)
    return subprocess.run([KAGGLE, *args], check=check)


def push_code():
    out = BUILD / "code"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(ROOT / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copytree(ROOT / "configs", out / "configs")
    (out / "data").mkdir()
    shutil.copy(ROOT / "data" / "processed" / "annotations.pkl", out / "data" / "annotations.pkl")
    (out / "dataset-metadata.json").write_text(
        json.dumps({"title": "gar-code", "id": CODE_DATASET, "licenses": [{"name": "CC0-1.0"}]}, indent=2)
    )

    exists = subprocess.run([KAGGLE, "datasets", "status", CODE_DATASET], capture_output=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(out), "-m", "update", "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(out), "--dir-mode", "zip")


def push_kernel(name, steps, reuse, timeout=None):
    out = BUILD / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    script = (ROOT / "kaggle" / "run.py").read_text(encoding="utf-8")
    if steps:
        script = re.sub(r"^STEPS = \[.*?\]\n", f"STEPS = {steps!r}\n", script, count=1, flags=re.S | re.M)
    (out / "run.py").write_text(script, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{USER}/{name}",
                "title": name,
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": SOURCES,
                "competition_sources": [],
                "kernel_sources": [f"{USER}/{k}" for k in reuse],
                "machine_shape": "NvidiaTeslaT4",
            },
            indent=2,
        )
    )
    kaggle(
        "kernels", "push", "-p", str(out), "--accelerator", "NvidiaTeslaT4", *(["-t", str(timeout)] if timeout else [])
    )


def main():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("action", choices=["code", "kernel", "status", "output"])
    parser.add_argument("--name", default="gar-pipeline", help="kernel slug")
    parser.add_argument("--steps", nargs="*", help="override STEPS in kaggle/run.py")
    parser.add_argument("--reuse", nargs="*", default=[], help="kernels whose output is attached as input")
    parser.add_argument("--timeout", type=int, help="seconds; Kaggle's own limit still applies")
    args = parser.parse_args()
    if args.action == "code":
        push_code()
    elif args.action == "kernel":
        push_kernel(args.name, args.steps, args.reuse, args.timeout)
    elif args.action == "status":
        kaggle("kernels", "status", f"{USER}/{args.name}")
    else:
        kaggle("kernels", "output", f"{USER}/{args.name}", "-p", str(ROOT / "results_kaggle" / args.name), "-o")


if __name__ == "__main__":
    main()
