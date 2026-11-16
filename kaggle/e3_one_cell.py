"""Run E3 on the mounted Kaggle inputs and save the report."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import bootstrap

BRATS = "dschettler8845/brats-2021-task1"
CHECKPOINT = "ahmadabbas176/232-ema-ckpt-pt/pytorch/default/1"


def main(args):
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(repo))
    bootstrap.install_missing_dependencies()

    import torch
    from experiments import e3

    gpu_count = min(torch.cuda.device_count(), 2)
    if not gpu_count:
        raise RuntimeError("Select a Kaggle GPU accelerator before running E3")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = visible.split(",")[:gpu_count] if visible else list(map(str, range(gpu_count)))
    mode = "smoke" if args.smoke else "full"
    root = Path("/tmp") / f"andi-e3-{mode}-{commit[:12]}"
    root.mkdir(exist_ok=True)
    output = Path("/kaggle/working") / f"andi-e3-{mode}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = bootstrap.mounted_checkpoint(CHECKPOINT)
    if e3.digest(checkpoint) != e3.CHECKPOINT_SHA:
        raise ValueError("Mounted checkpoint differs from the frozen ANDi checkpoint")
    transfer = args.ljubljana.resolve() if args.ljubljana else None
    design = e3.make_design(args.smoke, transfer is not None)
    config = {
        "identity": {"commit": commit, "brats": BRATS, "checkpoint": CHECKPOINT,
                     "checkpoint_sha256": e3.CHECKPOINT_SHA,
                     "ljubljana_dataset": str(transfer) if transfer and not args.smoke else None},
        "design": design,
    }
    design_path = root / "design.json"
    if design_path.exists() and e3.load_json(design_path) != config:
        raise ValueError(f"Saved experiment configuration differs in {root}")
    e3.e1.write_json(design_path, config)

    brats = bootstrap.mount(BRATS)

    def stage_brats(split, ids):
        destination = root / f"data-{split}"
        if not destination.exists():
            with TemporaryDirectory(dir=root) as temp:
                staged = Path(temp) / "data"
                bootstrap.stage_data(brats, ids, staged)
                staged.rename(destination)
        return destination

    def run_phase(phase, data):
        count = min(gpu_count, len(design[phase]))

        def run_one(shard):
            env = os.environ | {
                "CUDA_VISIBLE_DEVICES": gpu_ids[shard], "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "4", "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
            }
            command = [sys.executable, "-u", "-m", "experiments.e3", phase,
                       "--work", str(root), "--data", str(data), "--checkpoint", str(checkpoint),
                       "--shard", str(shard), "--shards", str(count)]
            subprocess.run(command, cwd=repo, env=env, check=True)

        print(f"E3 {phase}: {len(design[phase])} subjects on {count} GPU(s)", flush=True)
        with ThreadPoolExecutor(max_workers=count) as pool:
            list(pool.map(run_one, range(count)))
        e3.files_for(root, phase, design[phase])
        for shard in range(count):
            if e3.load_json(root / f"backend-{phase}-{shard}.json") != backend:
                raise ValueError("E3 requires the same GPU model and CUDA environment across shards")

    torch.set_num_threads(4)
    backend = e3.gpu_provenance()
    # Fit before opening evaluation images and preserve the reference with the output.
    validation = stage_brats("val", design["reference"] + design["validation"])
    run_phase("reference", validation)
    reference = e3.fit(root, config)
    e3.e1.write_json(output / "reference.json", reference)
    run_phase("validation", validation)
    if design["test"]:
        test = stage_brats("test", design["test"])
        run_phase("test", test)
    if design["ljubljana"]:
        run_phase("ljubljana", transfer)
    print("E3 scoring", flush=True)
    e3.report(root, output, config | {"backend": backend}, reference)
    print(f"E3 {mode} complete. See {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ljubljana", type=Path, help="Prepared registered Ljubljana data root")
    main(parser.parse_args())
