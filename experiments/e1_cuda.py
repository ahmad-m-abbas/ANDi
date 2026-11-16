"""Run the E1 phases independently on CUDA and join their saved artifacts."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import median_filter

from diffusion import Diffusion
from experiments import e1
from modules import UNet


CHECKPOINT_SHA = "ff76a1fadd74e839fea6d993f0b17b1e830d1a0bf127ef2f2e1e10afd92216bb"
MANIFEST = "andi-e1-artifact.json"
GPU_STAGES = ("smoke", "probe", "calibrate", "validate", "test", "noise")
STAGES = (*GPU_STAGES, "select", "score")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CudaInference(e1.Inference):
    def __init__(self, conf, split):
        if not torch.cuda.is_available():
            raise RuntimeError("This phase requires a CUDA GPU. Enable a GPU in Kaggle.")
        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(self.device)
        self.seed = conf["seed"]
        self.split = split
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model = UNet(device=self.device).to(self.device).eval()
        self.model.load_state_dict(torch.load(conf["model"], map_location="cpu", weights_only=True))
        self.diffusion = Diffusion(
            conf["noise_steps"], conf["size"], conf["beta_start"], conf["beta_end"], self.device,
        )

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def noise_states(self, batches, index):
        # Each draw has its own seed, so skipped timesteps require no RNG replay.
        return None

    def state_for(self, states, index, batch_index, step):
        split_offset = 1 << 40 if self.split == "test" else 0
        return split_offset + ((self.seed + index) * 1000 + step) * 32 + batch_index

    def set_noise_state(self, state):
        torch.cuda.manual_seed(state)


def setup(split, args):
    conf = yaml.safe_load(Path("conf/eval.yml").read_text())
    conf.update(path_to_csv=f"splits/BraTS21/scans_{split}.csv", workers=0,
                dataset_path=str(args.data), model=str(args.checkpoint))
    ids = e1.subject_ids(conf["path_to_csv"])
    if len(ids) != {"val": 62, "test": 251}[split] or len(set(ids)) != len(ids):
        raise ValueError(f"Unexpected BraTS21 {split} split")
    return conf, ids


def shard(ids, index, count):
    if not 0 <= index < count <= len(ids):
        raise ValueError("Shard indices are zero based and each shard must contain a subject")
    return ids[index::count]


def test_panel(ids):
    return [ids[i] for i in np.linspace(0, len(ids) - 1, e1.TIMING_PANEL_SIZE, dtype=int)]


def link(source, target):
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())


def collect(items, stage, location, ids, configs, panel=()):
    """Validate complete, non-overlapping shards before linking their files for scoring."""
    expected = set(ids)
    seen = set()
    for root, _ in items[stage]:
        directory = root / stage
        design = read(directory / "design.json")
        subjects = set(design["subjects"])
        if (not subjects or subjects & seen or subjects - expected
                or design["configs"] != configs
                or set(design["timing_panel"]) != subjects.intersection(panel)):
            raise ValueError(f"Incompatible or overlapping {stage} shard: {root}")
        seen |= subjects
        for subject in subjects:
            link(directory / "maps" / f"{subject}.npz", location / "maps" / f"{subject}.npz")
            if subject in panel:
                link(directory / f"{subject}.json", location / f"{subject}.json")
    if seen != expected:
        raise ValueError(f"Missing {stage} subjects: {sorted(expected - seen)}")
    e1.write_json(location / "design.json", dict(configs=configs, subjects=ids, timing_panel=list(panel)))


def single(items, stage):
    if len(items[stage]) != 1:
        raise ValueError(f"Expected exactly one {stage} artifact")
    return items[stage][0][0]


def stage_inputs(args, identity):
    dependencies = {
        "smoke": (), "probe": (), "calibrate": (), "validate": ("calibrate",),
        "select": ("calibrate", "validation"), "test": ("select",), "noise": ("select",),
        "score": ("select", "test", "noise") if getattr(args, "include_noise", False) else ("select", "test"),
    }[args.stage]
    items = {key: [] for key in dependencies}
    for path in args.input:
        metadata = read(path / MANIFEST)
        stage = metadata["stage"]
        if metadata["identity"] != identity or stage not in items:
            raise ValueError(f"Unexpected source or experiment identity: {path}")
        items[stage].append((path, metadata["backend"]))
    if any(not group for group in items.values()):
        raise ValueError(f"Required inputs for {args.stage}: {dependencies}")
    return items


def gpu_provenance():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")
    return dict(device="cuda", gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                checkpoint_sha256=CHECKPOINT_SHA,
                rng="CUDA Gaussian; seeded by split, original split row, timestep, and axial batch",
                hardware=f"{torch.cuda.get_device_name(0)} CUDA")


def check_backend(items, backend):
    if any(other != backend for group in items.values() for _, other in group):
        raise ValueError("GPU model or CUDA software differs between input artifacts")


def smoke(conf, inference):
    volume, _ = e1.MRI_Volume(conf).dataset[0]
    batches = inference.batches(volume.unsqueeze(0))
    torch.testing.assert_close(
        inference.residual(batches[0], 75, inference.state_for(None, 0, 0, 75)),
        inference.residual(batches[0], 75, inference.state_for(None, 0, 0, 75)),
        rtol=0, atol=0,
    )
    maps, seconds = inference.predict(batches, None, 0, {"smoke": (50, 75, 199)})
    if not np.isfinite(maps["smoke"]).all() or maps["smoke"].max() <= 0:
        raise ValueError("CUDA preflight produced an invalid score")
    e1.write_json(e1.OUTPUT / "smoke.json", {"timesteps": [50, 75, 199], "seconds": seconds})


def noise_check(conf, inference, steps, ids):
    dataset = e1.MRI_Volume(conf).dataset
    configs = {"baseline": list(e1.BASE), "selected_25": steps}
    subject_rows = list(range(len(ids)))
    trials = {}
    for seed in (conf["seed"], conf["seed"] + 1000):
        inference.seed = seed
        scores = {name: [] for name in configs}
        truth = []
        for row in subject_rows:
            volume, label = dataset[row]
            maps, _ = inference.predict(inference.batches(volume.unsqueeze(0)), None, row, configs)
            truth.append(label.numpy())
            for name, score in maps.items():
                scores[name].append(median_filter(score, size=5))
        labels = np.stack(truth)
        trials[str(seed)] = {}
        for name, values in scores.items():
            maps = np.stack(values)
            low, high = float(maps.min()), float(maps.max())
            dice = [e1.score_volume(scores, truth, low, high)[0] for scores, truth in zip(maps, labels)]
            ap = e1.pooled_average_precision(torch.from_numpy(maps), torch.from_numpy(labels))
            trials[str(seed)][name] = {"auprc": ap, "dice": float(np.mean(dice))}
    e1.write_json(e1.OUTPUT / "noise_seeds.json", dict(subjects=ids, steps=steps, trials=trials))


def probe(conf, inference, ids, backend):
    # A separate artifact from production. No official test data is opened.
    e1.infer_split(conf, inference, "val", "calibration", ids[:1],
                   {"baseline": e1.BASE}, provenance=backend)
    with np.load(e1.OUTPUT / "calibration/maps" / f"{ids[0]}.npz") as data:
        logs = np.maximum(data["logs"].reshape(len(e1.BASE), -1), np.log(1e-30))
    configs, groups = e1.candidates(inference, np.corrcoef(logs), contiguous=True)
    stage = e1.OUTPUT / "validation"
    e1.infer_split(conf, inference, "val", "validation", ids[1:2],
                   {"baseline": e1.BASE, **configs}, provenance=backend)
    with np.load(stage / "maps" / f"{ids[1]}.npz") as maps:
        if not all(np.isfinite(maps[name]).all() for name in configs):
            raise ValueError("The CUDA probe produced non-finite candidate scores")
        np.testing.assert_array_equal(maps["k25_t"], maps["w75_199"])
    e1.select_validation(ids[1:2], configs, groups, embedded_baseline=True)
    volume, _ = e1.MRI_Volume(conf).dataset[1]
    batches = inference.batches(volume.unsqueeze(0))
    actual = {}
    for name, steps in (("baseline", e1.BASE), ("k25_t", configs["k25_t"])):
        direct, seconds = inference.predict(batches, None, 1, {name: steps})
        np.testing.assert_allclose(direct[name], e1.saved_map(stage, ids[1], name), atol=1e-10, rtol=1e-5)
        actual[name] = seconds
    e1.write_json(e1.OUTPUT / "probe.json", {
        "calibration_subject": ids[0], "validation_subject": ids[1],
        "candidates": configs, "inference_seconds": actual,
    })


def run(args, identity):
    stage = args.stage
    items = stage_inputs(args, identity)
    split = "test" if stage in ("test", "score") else "val"
    conf, ids = setup(split, args)
    if stage in GPU_STAGES:
        if digest(args.checkpoint) != CHECKPOINT_SHA:
            raise ValueError("The EMA checkpoint hash differs from the frozen E1 checkpoint")
        backend = gpu_provenance()
        check_backend(items, backend)
        inference = CudaInference(conf, split)
    else:
        primary = "calibrate" if stage == "select" else "select"
        backend = items[primary][0][1]
        check_backend(items, backend)
    e1.write_json(e1.OUTPUT / "provenance.json", {**backend, "identity": identity})

    if stage == "smoke":
        with torch.inference_mode():
            smoke(conf, inference)
        return backend, ["smoke.json", "provenance.json"]
    if stage == "probe":
        probe(conf, inference, ids, backend)
        return backend, ["probe.json", "provenance.json", "frozen.json", "validation/metrics"]
    if stage == "calibrate":
        correlation = e1.calibrate(conf, inference, provenance=backend)
        configs, groups = e1.candidates(inference, correlation, contiguous=True)
        path = e1.OUTPUT / "calibration" / "candidates.json"
        e1.write_json(path, dict(configs=configs, groups=groups))
        return backend, ["calibration/diagnostics.json", "calibration/candidates.json"]

    if stage in ("validate", "select"):
        calibration = single(items, "calibrate") / "calibration"
        design = read(calibration / "candidates.json")
        configs = {"baseline": list(e1.BASE), **design["configs"]}
        ids = ids[e1.CALIBRATION_SIZE:]
        if stage == "validate":
            subset = shard(ids, args.shard, args.shards)
            e1.infer_split(conf, inference, split, "validation", subset, configs, provenance=backend)
            return backend, ["validation"]
        collect(items, "validation", e1.OUTPUT / "validation", ids, configs)
        e1.select_validation(ids, design["configs"], design["groups"], embedded_baseline=True)
        shutil.copytree(calibration, e1.OUTPUT / "calibration")
        return backend, ["frozen.json", "calibration", "validation/design.json", "validation/metrics"]

    selection = single(items, "select")
    frozen = read(selection / "frozen.json")
    if stage == "noise":
        noise_check(conf, inference, frozen["configs"][frozen["winners"]["k25"]], ids[:2])
        return backend, ["noise_seeds.json"]
    if set(ids) & set(frozen["validation_subjects"]) or frozen["baseline"] != list(e1.BASE):
        raise ValueError("Test split or frozen baseline is invalid")
    configs = {"baseline": list(e1.BASE), **frozen["configs"]}
    panel = test_panel(ids)
    freeze_sha = digest(selection / "frozen.json")
    if stage == "test":
        if args.timing_only:
            subset = panel
        else:
            eligible = [subject for subject in ids if subject not in panel] if args.exclude_timing else ids
            subset = shard(eligible, args.shard, args.shards)
        e1.infer_split(conf, inference, split, "test", subset, configs,
                       [subject for subject in panel if subject in subset], provenance=backend)
        e1.write_json(e1.OUTPUT / "test" / "freeze.json", {"sha256": freeze_sha})
        return backend, ["test"]

    for root, _ in items["test"]:
        if read(root / "test" / "freeze.json")["sha256"] != freeze_sha:
            raise ValueError("Test shard used a different validation freeze")
    collect(items, "test", e1.OUTPUT / "test", ids, configs, panel)
    for relative in ("frozen.json", "calibration", "validation"):
        source, target = selection / relative, e1.OUTPUT / relative
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)
    e1.score_test(ids, frozen["configs"], panel, embedded_baseline=True)
    if "noise" in items:
        shutil.copyfile(single(items, "noise") / "noise_seeds.json", e1.OUTPUT / "noise_seeds.json")
    from experiments.e1_report import main as report

    report(embedded_baseline=True)
    outputs = ["frozen.json", "provenance.json", "calibration", "validation", "results.json",
               "report.md", "diagnostics.png", "pareto_mf5.png", "example.png"]
    if "noise" in items:
        outputs.append("noise_seeds.json")
    return backend, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--identity", required=True, type=json.loads)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", action="append", type=Path, default=[])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    timings = parser.add_mutually_exclusive_group()
    timings.add_argument("--timing-only", action="store_true")
    timings.add_argument("--exclude-timing", action="store_true")
    parser.add_argument("--include-noise", action="store_true")
    args = parser.parse_args()
    if args.stage in GPU_STAGES and (args.data is None or args.checkpoint is None):
        parser.error("GPU stages require --data and --checkpoint")
    e1.OUTPUT = Path("results/brats21-e1-cuda")
    e1.STEP_CACHE = Path("results/cache/e1-cuda")
    e1.OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    backend, outputs = run(args, args.identity)
    args.output.mkdir(parents=True, exist_ok=True)
    for relative in outputs:
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(e1.OUTPUT / relative, destination)
    if e1.STEP_CACHE.exists():
        shutil.move(e1.STEP_CACHE, args.output / "step-cache")
    stage = "validation" if args.stage == "validate" else args.stage
    e1.write_json(args.output / MANIFEST, dict(stage=stage, backend=backend, identity=args.identity))


if __name__ == "__main__":
    main()
