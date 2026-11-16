"""Generate a private Kaggle script job from the current committed checkout."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from bootstrap import GPU_STAGES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*GPU_STAGES, "select", "score"))
    parser.add_argument("--id", required=True, help="Kaggle owner/kernel-slug")
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint-model", help="Pinned owner/model/framework/variation/version")
    checkpoint.add_argument("--checkpoint-dataset", help="Private owner/dataset-slug")
    parser.add_argument("--brats-dataset", default="dschettler8845/brats-2021-task1")
    parser.add_argument("--kernel-source", action="append", default=[])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo", default="https://github.com/ahmad-m-abbas/ANDi.git")
    parser.add_argument("--branch", default="codex/stack-05-e3")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int)
    args = parser.parse_args()
    all_ids = [args.id, args.brats_dataset, *args.kernel_source]
    if args.checkpoint_dataset:
        all_ids.append(args.checkpoint_dataset)
    if any(not re.fullmatch(r"[\w-]+/[\w-]+", ref) for ref in all_ids):
        parser.error("Kaggle IDs must use owner/slug")
    if args.checkpoint_model and not re.fullmatch(r"(?:[\w-]+/){4}[1-9][0-9]*", args.checkpoint_model):
        parser.error("Checkpoint model must use owner/model/framework/variation/version")
    if len(set(args.kernel_source)) != len(args.kernel_source):
        parser.error("List each upstream kernel once")
    count = args.shards if args.shards is not None else {"validate": 3, "test": 8}.get(args.stage, 1)
    if not 0 <= args.shard < count:
        parser.error("Use a zero-based shard index smaller than the shard count")
    if args.stage not in ("validate", "test") and count != 1:
        parser.error("Only validation and test inference can be sharded")
    inputs = len(args.kernel_source)
    required = {"validate": 1, "test": 1, "noise": 1}.get(args.stage, 0)
    if args.stage in ("select", "score"):
        if inputs < 2:
            parser.error("Scoring requires a parent stage and the inference shards")
    elif inputs != required:
        parser.error(f"{args.stage} requires {required} upstream kernel source(s)")

    gpu = args.stage in GPU_STAGES
    checkpoint_ref = args.checkpoint_model or args.checkpoint_dataset
    datasets = [args.brats_dataset] if gpu else []
    if gpu and args.checkpoint_dataset:
        datasets.append(args.checkpoint_dataset)
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    settings = {
        "stage": args.stage, "shard": args.shard, "shards": count,
        "repo": args.repo, "branch": args.branch,
        "kernel_sources": args.kernel_source,
        "identity": {"commit": commit, "brats": args.brats_dataset, "checkpoint": checkpoint_ref},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    code_file = "run_e1.py"
    bootstrap = Path(__file__).with_name("bootstrap.py").read_text()
    (args.out / code_file).write_text(bootstrap + f"\n\nif __name__ == '__main__':\n    main({settings!r})\n")
    metadata = {
        "id": args.id,
        "title": args.id.split("/")[1].replace("-", " "),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": datasets,
        "model_sources": [args.checkpoint_model] if gpu and args.checkpoint_model else [],
        "kernel_sources": args.kernel_source,
    }
    (args.out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {args.id} at {commit} in {args.out}")


if __name__ == "__main__":
    main()
