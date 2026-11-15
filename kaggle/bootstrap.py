"""Kernel entry point. prepare.py inserts the job settings ahead of execution."""

import csv
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/andi-e1")
ARTIFACT = "andi-e1-artifact.json"
MODALITIES = ("flair", "t1", "t1ce", "t2", "seg")
GPU_STAGES = ("smoke", "calibrate", "validate", "test")


def mount(input_id):
    slug = input_id.split("/")[-1]
    matches = [path for path in INPUT.rglob(slug) if path.is_dir()]
    if len(matches) != 1:
        raise ValueError(f"Expected one Kaggle mount for {input_id}: {matches}")
    return matches[0]


def one_file(root, name):
    files = sorted(root.rglob(name))
    if len(files) != 1:
        raise ValueError(f"Expected one {name} under {root}: {files}")
    return files[0]


def checkout(settings, repo):
    commit = settings["identity"]["commit"]
    subprocess.run(
        ["git", "clone", "--single-branch", "--branch", settings["branch"], "--no-tags",
         settings["repo"], str(repo)],
        check=True, env=os.environ | {"GIT_TERMINAL_PROMPT": "0"},
    )
    subprocess.run(["git", "checkout", "--detach", commit], cwd=repo, check=True)


def read_ids(repo, settings):
    stage = settings["stage"]
    split = "test" if stage == "test" else "val"
    with (repo / f"splits/BraTS21/scans_{split}.csv").open() as stream:
        ids = [row["BraTS21ID"] for row in csv.DictReader(stream)]
    if stage == "smoke":
        return ids[:1]
    if stage == "calibrate":
        return ids[:8]
    if stage == "validate":
        ids = ids[8:]
    return ids[settings["shard"]::settings["shards"]]


def stage_data(root, ids, destination):
    """Expose the five original NIfTI files per selected subject to MRI_Volume."""
    names = {f"{subject}_{suffix}.nii.gz": subject for subject in ids for suffix in MODALITIES}
    for directory in root.rglob("BraTS2021_*"):
        if not directory.is_dir() or directory.name not in ids:
            continue
        for suffix in MODALITIES:
            name = f"{directory.name}_{suffix}.nii.gz"
            source = directory / name
            if name in names and source.is_file():
                target = destination / names.pop(name) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve())
    if not names:
        return
    for path in sorted(p for p in root.rglob("*") if p.name.endswith((".tar", ".tar.gz", ".tgz"))):
        with tarfile.open(path, mode="r|*") as archive:
            for item in archive:
                name = PurePosixPath(item.name).name
                if item.isfile() and name in names:
                    target = destination / names.pop(name) / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(item) as source, target.open("wb") as stream:
                        shutil.copyfileobj(source, stream)
                    if not names:
                        return
    raise FileNotFoundError(f"Missing {len(names)} required BraTS21 NIfTI files, e.g. {sorted(names)[:5]}")


def main(settings):
    with TemporaryDirectory(prefix="andi-e1-") as work:
        repo = Path(work) / "ANDi"
        checkout(settings, repo)
        required = ("lmdb", "nibabel", "accelerate", "numba")
        missing = [package for package in required if importlib.util.find_spec(package) is None]
        if missing:
            subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
        output = Path(work) / "artifacts"
        stage = settings["stage"]
        command = [
            sys.executable, "-u", "-m", "experiments.e1_cuda", stage,
            "--identity", json.dumps(settings["identity"]), "--output", str(output),
            "--shard", str(settings["shard"]), "--shards", str(settings["shards"]),
        ]
        if stage in GPU_STAGES:
            ids = read_ids(repo, settings)
            dataset = Path(work) / "BraTS21"
            stage_data(mount(settings["identity"]["brats"]), ids, dataset)
            checkpoint = one_file(mount(settings["identity"]["checkpoint"]), "232_ema_ckpt.pt")
            command.extend(("--data", str(dataset), "--checkpoint", str(checkpoint)))
        for kernel in settings["kernel_sources"]:
            command.extend(("--input", str(one_file(mount(kernel), ARTIFACT).parent)))
        env = os.environ | {
            "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        print(f"Running {stage} on commit {settings['identity']['commit']}", flush=True)
        subprocess.run(command, cwd=repo, env=env, check=True)
        total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if total > 19_000_000_000:
            raise RuntimeError(f"Stage produced {total} bytes. Use a larger shard count to fit Kaggle output.")
        shutil.move(output, OUTPUT)
        print(f"Saved {total:,} bytes to {OUTPUT}", flush=True)
