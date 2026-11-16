"""Run the full validation-first E1 or a two-subject preflight in one Kaggle job."""

import argparse
import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bootstrap


def full(commit, probe=False):
    import torch

    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    available = torch.cuda.device_count()
    if not available:
        raise RuntimeError("Select a Kaggle GPU accelerator before running E1")
    gpu_ids = devices.split(",")[:available] if devices else list(map(str, range(available)))
    scratch = Path("/tmp") / f"andi-e1-{'probe' if probe else 'full'}-{commit[:12]}"
    scratch.mkdir(exist_ok=True)
    saved = Path("/kaggle/working") / ("andi-e1-probe" if probe else "andi-e1-full")
    saved.mkdir(parents=True, exist_ok=True)
    settings = {
        "repo": "https://github.com/ahmad-m-abbas/ANDi.git",
        "branch": "codex/stack-05-e3",
        "kernel_sources": [],
        "identity": {
            "commit": commit,
            "brats": "dschettler8845/brats-2021-task1",
            "checkpoint": "ahmadabbas176/232-ema-ckpt-pt/pytorch/default/1",
        },
    }

    def stage(name, kind, parents=(), shard=0, shards=1, data=None, gpu=gpu_ids[0], flags=()):
        output = scratch / name
        if not (output / bootstrap.ARTIFACT).exists():
            bootstrap.main(
                {**settings, "stage": kind, "shard": shard, "shards": shards},
                output_path=output, local_inputs=[scratch / item for item in parents],
                staged_data=data, gpu=gpu, extra_args=flags,
            )
        cache = output / "step-cache"
        if cache.is_dir():
            shutil.move(cache, saved / f"{name}-step-cache")
        return output

    if probe:
        artifact = stage("probe", "probe")
        shutil.copytree(artifact, saved / "report", dirs_exist_ok=True)
        print(f"Preflight complete. See {saved / 'report/probe.json'}", flush=True)
        return

    source = Path(__file__).resolve().parents[1]
    brats = bootstrap.mount(settings["identity"]["brats"])

    def prepare_data(split):
        destination = scratch / f"data-{split}"
        ready = scratch / f"data-{split}.ready"
        if not ready.exists():
            with (source / f"splits/BraTS21/scans_{split}.csv").open() as stream:
                ids = [row["BraTS21ID"] for row in csv.DictReader(stream)]
            bootstrap.stage_data(brats, ids, destination)
            ready.touch()
        return destination

    def infer_shards(kind, count, parents, data, flags=()):
        names = [f"{kind}-{i}" for i in range(count)]

        def worker(worker_index, gpu):
            for i in range(worker_index, count, len(gpu_ids)):
                stage(names[i], kind, parents, i, count, data, gpu, flags)

        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = [executor.submit(worker, i, gpu) for i, gpu in enumerate(gpu_ids)]
            for future in futures:
                future.result()
        return names

    validation = prepare_data("val")
    stage("calibrate", "calibrate", data=validation)
    validation_shards = infer_shards("validate", 4, ["calibrate"], validation)
    selection = stage("select", "select", ["calibrate", *validation_shards])
    shutil.copy2(selection / "frozen.json", saved / "frozen.json")
    stage("noise", "noise", ["select"], data=validation)
    test = prepare_data("test")
    test_shards = infer_shards("test", 8, ["select"], test, ["--exclude-timing"])
    # Time each frozen method on an otherwise idle GPU and host.
    stage("timing", "test", ["select"], data=test, flags=["--timing-only"])
    report = stage("score", "score", ["select", *test_shards, "timing", "noise"], flags=["--include-noise"])
    shutil.copytree(report, saved / "report", dirs_exist_ok=True)
    print(f"E1 complete. See {saved / 'report/report.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("--probe", action="store_true")
    options = parser.parse_args()
    full(options.commit, options.probe)
