"""Prepare fixed-split healthy slices and run E4 or its controls on one device."""

import argparse
import hashlib
import math
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

if __package__:
    from . import bootstrap
else:
    # The notebook executes this file directly, with its directory on sys.path.
    import bootstrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BRATS = "dschettler8845/brats-2021-task1"
CHECKPOINT = "ahmadabbas176/232-ema-ckpt-pt/pytorch/default/1"
CHECKPOINT_SHA = "ff76a1fadd74e839fea6d993f0b17b1e830d1a0bf127ef2f2e1e10afd92216bb"


def prepare(repo, source, work, split, smoke):
    import lmdb
    import numpy as np
    import pandas as pd
    from torchvision import transforms
    from split_healthy import healthy_slice_batch
    from andi.data.loading import LMDBDataset

    split_path = repo / f"splits/BraTS21/scans_{split}.csv"
    subjects = pd.read_csv(split_path).iloc[:, 0].tolist()
    if smoke:
        subjects = subjects[:1]
    staged = work / split
    bootstrap.stage_data(source, subjects, staged)
    destination = work / f"lmdb-{split}"
    records = []
    content = hashlib.sha256()
    count = 0
    # Sparse virtual map, sized above the maximum 155 slices per training subject.
    map_size = max(64 * 1024**2, len(subjects) * 155 * 4 * 128 * 128 * 4 * 2)
    resize = transforms.Resize(128, antialias=True)
    slice_limit = {"train": 8, "val": 4}[split] if smoke else None
    with lmdb.open(str(destination), map_size=map_size) as env:
        for subject in subjects:
            indices, images = healthy_slice_batch(staged, subject, resize, slice_limit)
            if images is not None:
                if not np.isfinite(images).all():
                    raise ValueError(f"Nonfinite input from {subject}")
                with env.begin(write=True) as txn:
                    for index, image in zip(indices, images):
                        txn.put(f"{count:08}".encode(), pickle.dumps(image))
                        content.update(image.tobytes())
                        records.append((subject, index))
                        count += 1
            # stage_data only creates symlinks or extracted copies inside our work area.
            shutil.rmtree(staged / subject)
    shutil.rmtree(staged)
    if count < 2:
        raise ValueError(f"Too few healthy {split} slices")
    pd.DataFrame(records, columns=["BraTS21ID", "Slice"]).to_csv(
        work / f"healthy-{split}.csv", index=False
    )
    identity = {
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "subjects": subjects, "slice_count": count,
        "slice_index_sha256": hashlib.sha256(json.dumps(records).encode()).hexdigest(),
        "tensor_sha256": content.hexdigest(),
    }
    return LMDBDataset(str(destination), transforms.Compose([])), identity


def assert_same(left, right):
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_same(first, second)
    else:
        assert left == right, (left, right)


def smoke_resume(trainer, store, remote, config, training, validation, device, run):
    """Compare an uninterrupted next update with a saved/reloaded update."""
    import copy
    import torch
    from andi.training.e4 import Trainer, restore_rng

    trainer.save(store)
    trainer.report(run, trainer.step())
    path, ref = trainer.save(store)
    # The snapshot is loaded independently from durable storage when online.
    if remote:
        _, state = remote.recover(ref)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    restore_rng(state["rng"])
    trainer.step()
    expected = copy.deepcopy(trainer.snapshot())
    del trainer
    resumed = Trainer(config, training, validation, device)
    resumed.restore(state)
    metrics = resumed.step()
    assert_same(expected, resumed.snapshot())
    del expected
    print("PASS exact next-update save/reload/resume comparison", flush=True)
    resumed.report(run, metrics | {"resume_check": True})
    resumed.fit(store, run, save_steps=1, save_seconds=600)
    return resumed


def main(args):
    repo = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    bootstrap.install_missing_dependencies()
    import numpy as np
    import torch
    import torchvision
    import yaml
    from andi.training.e4 import Trainer
    from andi.training.e4_checkpoints import CheckpointStore, WandbCheckpoints, sha256

    if args.local_only and not args.smoke:
        raise ValueError("--local-only is restricted to smoke checks")
    if not args.local_only and (not args.entity or not args.project):
        raise ValueError("W&B entity and project are required for durable checkpoints")
    if args.resume and args.initialize:
        raise ValueError("Choose exact resume or pretrained initialization")
    if not args.local_only:
        bootstrap.install_missing_dependencies(("wandb",))
    if not args.local_only and not args.data and not torch.cuda.is_available():
        raise RuntimeError("Select one Kaggle GPU for the online training cell")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Use a deterministic attention path for exact same-device smoke comparisons.
    if device.startswith("cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        # Fail early on accelerator builds without deterministic bilinear backward.
        probe = torch.zeros((1, 1, 2, 2), device=device, requires_grad=True)
        torch.nn.functional.interpolate(
            probe, scale_factor=2, mode="bilinear", align_corners=True
        ).sum().backward()

    output = args.output.resolve() / uuid4().hex[:12]
    output.mkdir(parents=True)
    run = None
    if not args.local_only:
        import wandb
        run = wandb.init(entity=args.entity, project=args.project, mode="online",
                         job_type="e4-smoke" if args.smoke else "e4-training",
                         dir=str(output), name=args.collection)
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")
    try:
        remote = WandbCheckpoints(run, args.collection) if run else None
        saved = None
        resolved = None
        if args.resume:
            if not remote:
                raise ValueError("Use online mode to restore a verified W&B checkpoint")
            prefix = f"{args.entity}/{args.project}/{args.collection}:"
            if not args.resume.startswith(prefix):
                raise ValueError("Resume must use the same entity/project/collection")
            resolved, saved = remote.recover(args.resume)

        with TemporaryDirectory(prefix="andi-e4-") as temporary:
            work = Path(temporary)
            source = args.data.resolve() if args.data else bootstrap.mount(BRATS)
            training, train_identity = prepare(repo, source, work, "train", args.smoke)
            validation, val_identity = prepare(repo, source, work, "val", args.smoke)
            for split in ("train", "val"):
                shutil.copyfile(work / f"healthy-{split}.csv", output / f"healthy-{split}.csv")
            config = yaml.safe_load((repo / "conf/train.yml").read_text())
            config.update(yaml.safe_load((repo / "conf/e4.yml").read_text()))
            config["variant"] = args.variant
            if args.variant != "consistency":
                config["consistency_lambda"] = 0.
            elif args.consistency_lambda is not None:
                config["consistency_lambda"] = args.consistency_lambda
            if not 0 <= config["consistency_lambda"] < math.inf:
                raise ValueError("Consistency weight must be nonnegative and finite")
            if args.smoke:
                config.update(epochs=1, batch_size=2, microbatch_size=1, warmup_steps=0.25)
            else:
                if args.epochs is not None:
                    config["epochs"] = args.epochs
                if args.batch_size is not None:
                    config["batch_size"] = args.batch_size
                if args.microbatch_size is not None:
                    config["microbatch_size"] = args.microbatch_size
                if config["microbatch_size"] > config["batch_size"]:
                    raise ValueError("Microbatch size must be at most the effective batch size")
            if config["augmentation_type"] != "gamma" or not 0 <= config["augmentation_magnitude"] < 1:
                raise ValueError("E4 uses gamma augmentation with magnitude in [0, 1)")
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            initial_weights = None
            initialization = {"mode": "fresh"}
            if saved:
                initialization = saved["config"]["initialization"]
            elif args.initialize:
                checkpoint = bootstrap.mounted_checkpoint(CHECKPOINT)
                if sha256(checkpoint) != CHECKPOINT_SHA:
                    raise ValueError("Pretrained input differs from the pinned ANDi checkpoint")
                initial_weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
                initialization = {"mode": "pretrained", "checkpoint": CHECKPOINT,
                                  "sha256": CHECKPOINT_SHA}
            config.update(
                initialization=initialization, smoke=args.smoke,
                identity={"commit": commit, "dataset": BRATS, "train": train_identity,
                          "validation": val_identity},
                backend={"python": platform.python_version(), "torch": str(torch.__version__),
                         "torchvision": str(torchvision.__version__), "numpy": np.__version__,
                         "cuda": torch.version.cuda,
                         "cudnn": torch.backends.cudnn.version(),
                         "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"},
                selection="minimum epoch EMA healthy-validation noise-prediction MSE",
            )
            # Machine-local paths and legacy continuation placeholders are unused.
            for key in ("dataset_path", "model", "ema_model", "opt", "train_continue",
                        "run_name", "workers", "horizontal_flip", "start_ckpt", "save_ckpt", "num_images"):
                config.pop(key, None)
            (output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
            (output / "resume.json").write_text(json.dumps(
                {"requested": args.resume, "resolved": resolved,
                 "save_steps": args.save_steps, "save_seconds": args.save_seconds}, indent=2) + "\n")
            if run:
                run.config.update(config | {"resume_artifact": resolved,
                                            "save_steps": args.save_steps,
                                            "save_seconds": args.save_seconds})
            trainer = Trainer(config, training, validation, device, initial_weights)
            if saved:
                trainer.restore(saved)
                del saved
            store = CheckpointStore(output / "checkpoints", remote)
            if args.smoke and trainer.global_step == 0:
                trainer = smoke_resume(trainer, store, remote, config, training, validation, device, run)
            else:
                trainer.fit(store, run, save_steps=args.save_steps, save_seconds=args.save_seconds)
            # Existing evaluators continue to receive a plain EMA state dict.
            best_path = store.directory / "best.pt"
            selected_ref = None
            if remote:
                with TemporaryDirectory() as best_work:
                    artifact, downloaded = remote.download(
                        f"{args.entity}/{args.project}/{args.collection}:best-step-{trainer.best_step}", best_work
                    )
                    if f"verified-{artifact.version}" not in artifact.aliases:
                        raise ValueError("Best checkpoint is unverified")
                    selected_ref = f"{args.entity}/{args.project}/{args.collection}:{artifact.version}"
                    best = torch.load(downloaded, map_location="cpu", weights_only=False)
            else:
                best = torch.load(best_path, map_location="cpu", weights_only=False)
            if (best["config"] != config or best["global_step"] != trainer.best_step
                    or best["best_validation"] != trainer.best_validation):
                raise ValueError("Best checkpoint differs from the recorded validation selection")
            torch.save(best["ema_model"], output / "selected_ema_ckpt.pt")
            selection = {"checkpoint": selected_ref, "global_step": trainer.best_step,
                         "validation_mse": trainer.best_validation}
            (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
            if run:
                run.summary["selected_checkpoint"] = selection
            print(f"E4 completed. Outputs at {output}", flush=True)
    finally:
        if run:
            run.finish(exit_code=int(sys.exc_info()[0] is not None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--collection", required=True, help="Unique stable name per variant/seed/configuration")
    parser.add_argument("--variant", choices=("baseline", "augmentation", "consistency"), default="consistency")
    parser.add_argument("--consistency-lambda", type=float)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--resume", help="Trusted entity/project/collection:verified or :vN")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Effective batch size per optimizer update")
    parser.add_argument("--microbatch-size", type=int, help="Slices processed together on device")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-seconds", type=float, default=600)
    parser.add_argument("--data", type=Path, help="Local extracted BraTS21 root for developer smoke checks")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/andi-e4"))
    parser.add_argument("--local-only", action="store_true", help="Explicit local smoke check without durability")
    arguments = parser.parse_args()
    if arguments.save_steps < 1 or not 0 < arguments.save_seconds < math.inf:
        parser.error("Checkpoint step and time intervals must be positive and finite")
    if any(value is not None and value < 1
           for value in (arguments.epochs, arguments.batch_size, arguments.microbatch_size)):
        parser.error("Epochs and batch sizes must be positive")
    main(arguments)
