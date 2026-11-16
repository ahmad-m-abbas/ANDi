"""BraTS21 timestep ablation using the archived baseline's per-subject MPS RNG."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from scipy.ndimage import binary_dilation, generate_binary_structure, median_filter
from sklearn.metrics import average_precision_score
from skimage.filters import threshold_yen

from diffusion import Diffusion
from eval import pooled_average_precision
from experiments.e2_cached import CEILING_THRESHOLDS, threshold_dice
from modules import UNet
from utils import MRI_Volume


BASE = tuple(range(75, 200))
BATCH = 8
SHAPE = (128, 128, 155)
OUTPUT = Path("results/brats21-e1")
STEP_CACHE = Path("results/cache/e1")
BUDGETS = (10, 25, 50, 75)
WINDOWS = ((50, 124), (75, 149), (75, 199), (125, 199), (100, 224), (50, 249))
RANDOM_SEEDS = (73, 74, 75)
CALIBRATION_SIZE = 8
TIMING_PANEL_SIZE = 8
FILTERS = (0, 3, 5)
DILATION = generate_binary_structure(3, 1)
ARCHIVE = {
    "val": Path("results/reproduction/maps_val"),
    "test": Path("results/brats21-mps/saved-baseline/maps"),
}


def write_json(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def write_npz(path, compressed=False, **arrays):
    pending = path.with_suffix(".tmp")
    with pending.open("wb") as stream:
        save = np.savez_compressed if compressed else np.savez
        save(stream, **arrays)
    pending.replace(path)


def subject_ids(path):
    with open(path) as source:
        return [row["BraTS21ID"] for row in csv.DictReader(source)]


def archived(split, subject):
    with np.load(ARCHIVE[split] / f"{subject}.npz", allow_pickle=False) as data:
        return data["score"], data["label"]


class Inference:
    def __init__(self, conf):
        if not torch.backends.mps.is_available():
            raise RuntimeError("Metal GPU is required")
        self.device = torch.device("mps")
        self.seed = conf["seed"]
        self.model = UNet(device=self.device).to(self.device).eval()
        self.model.load_state_dict(torch.load(conf["model"], map_location="cpu", weights_only=True))
        self.diffusion = Diffusion(
            conf["noise_steps"], conf["size"], conf["beta_start"], conf["beta_end"], self.device
        )

    def batches(self, volume):
        slices = ((volume[0] * 2) - 1).permute(3, 0, 1, 2).contiguous()
        return [batch.to(self.device) for batch in slices.split(BATCH)]

    def noise_states(self, batches, index):
        """Record original pre-draw MPS states in historical slice/timestep order."""
        torch.manual_seed(self.seed + index)
        states = []
        for batch in batches:
            batch_states = {}
            for step in reversed(BASE):
                batch_states[step] = torch.mps.get_rng_state()
                torch.randn_like(batch)
            states.append(batch_states)
        torch.mps.synchronize()
        return states

    def residual(self, images, step, state):
        torch.mps.set_rng_state(state)
        t = torch.full((len(images),), step, dtype=torch.long, device=self.device)
        x_t, noise = self.diffusion.noise_images(images, t)
        prediction = self.model(x_t, t)
        return (self.diffusion.ddpm_mu_t(x_t, noise, t)
                - self.diffusion.ddpm_mu_t(x_t, prediction, t)).square()

    def state_for(self, states, index, batch_index, step):
        if step in states[batch_index]:
            return states[batch_index][step]
        # Independent extension of the archived RNG stream, keyed by split-row,
        # step, and slice batch. In-window draws retain the archived RNG.
        seed = (1 << 40) + ((self.seed + index) * 1000 + step) * 32 + batch_index
        torch.mps.manual_seed(seed)
        return torch.mps.get_rng_state()

    @torch.inference_mode()
    def predict(self, batches, states, index, configs, observe=None):
        steps = sorted({step for config in configs.values() for step in config})
        offsets = {step: col for col, step in enumerate(steps)}
        columns = {key: [offsets[t] for t in ts] for key, ts in configs.items()}
        maps = {name: np.empty(SHAPE, np.float32) for name in configs}
        torch.mps.synchronize()
        start = perf_counter()
        for batch_index, images in enumerate(batches):
            residuals = torch.empty((len(images), len(steps), *images.shape[1:]), device=self.device)
            for step in reversed(steps):
                state = self.state_for(states, index, batch_index, step)
                residuals[:, offsets[step]] = self.residual(images, step, state)
            logs = residuals.cpu().log_()
            del residuals
            first = batch_index * BATCH
            last = first + len(images)
            if observe is not None:
                observe(steps, logs, first, last)
            for name, cols in columns.items():
                chosen = logs if len(cols) == len(steps) else logs[:, cols]
                result = chosen.mean(dim=1).exp_().amax(dim=1).permute(1, 2, 0)
                maps[name][:, :, first:last] = result.numpy()
        torch.mps.synchronize()
        return maps, perf_counter() - start


def verify(conf, inference):
    split = "val"
    ids = subject_ids(conf["path_to_csv"])
    volume, label = next(iter(MRI_Volume(conf)))
    batches = inference.batches(volume)
    states = inference.noise_states(batches, 0)
    maps, elapsed = inference.predict(batches, states, 0, {"baseline": BASE})
    score, truth = archived(split, ids[0])
    if not np.array_equal(label[0].numpy(), truth):
        raise ValueError("Archived segmentation differs from baseline preprocessing")
    delta = np.abs(maps["baseline"] - score)
    metrics = {"subject": ids[0], "seconds": elapsed, "max_abs": float(delta.max()),
               "max_rel": float((delta / np.maximum(score, 1e-12)).max())}
    write_json(OUTPUT / "replay_check.json", metrics)
    print(metrics, flush=True)
    np.testing.assert_allclose(maps["baseline"], score, rtol=1e-5, atol=1e-10)


def spaced(start, end, budget, coordinate):
    available = list(range(start, end + 1))
    selected = []
    for target in np.linspace(coordinate[start], coordinate[end], budget):
        closest = min(available, key=lambda i: (abs(coordinate[i] - target), i))
        selected.append(closest)
        available.remove(closest)
    return sorted(selected)


def timing_pilot(conf, inference):
    """Check the reduced-step path without accessing the held-out split."""
    subject = subject_ids(conf["path_to_csv"])[0]
    volume, label = MRI_Volume(conf).dataset[0]
    batches = inference.batches(volume.unsqueeze(0))
    states = inference.noise_states(batches, 0)
    steps = spaced(BASE[0], BASE[-1], 25, np.arange(conf["noise_steps"]))
    results = {}
    for name, selected in (("baseline", BASE), ("uniform_t_25", steps)):
        maps, seconds = inference.predict(batches, states, 0, {name: selected})
        score = maps[name]
        if name == "baseline":
            np.testing.assert_array_equal(score, archived("val", subject)[0])
        filtered = median_filter(score, size=5)
        truth = label.numpy()
        dice, _ = score_volume(filtered, truth, filtered.min(), filtered.max())
        ap = float(average_precision_score(truth.ravel(), filtered.ravel()))
        results[name] = dict(timesteps=selected, inference_s=seconds, auprc=ap, dice=dice)
        print(f"{name}: {seconds:.2f} s, AP={ap:.6f}, Dice={dice:.6f}", flush=True)
    speedup = results["baseline"]["inference_s"] / results["uniform_t_25"]["inference_s"]
    write_json(OUTPUT / "validation_timing_pilot.json", dict(
        subject=subject, results=results, inference_speedup=speedup,
    ))
    print(f"Validation timing pilot: {speedup:.2f}× speedup", flush=True)


def diverse(correlation, budget):
    cost = np.abs(correlation.copy())
    np.fill_diagonal(cost, np.inf)
    pair = np.unravel_index(cost.argmin(), cost.shape)
    selected = list(pair)
    while len(selected) < budget:
        selected.append(int(cost[selected].sum(axis=0).argmin()))
    return sorted(BASE[t] for t in selected)


def coordinates(inference):
    alpha = inference.diffusion.alpha_hat.cpu().numpy().astype(np.float64)
    return dict(t=np.arange(len(alpha)), noise=1 - alpha, log_snr=np.log(alpha / (1 - alpha)))


def candidates(inference, correlation):
    spaces = coordinates(inference)
    configs = {}
    families = {}
    for budget in BUDGETS:
        family = {}
        for key, coordinate in spaces.items():
            family[f"k{budget}_{key}"] = spaced(75, 199, budget, coordinate)
        family[f"k{budget}_diverse"] = diverse(correlation, budget)
        for seed in RANDOM_SEEDS:
            sample = np.random.default_rng(seed).choice(BASE, size=budget, replace=False)
            family[f"k{budget}_random{seed}"] = sorted(sample.tolist())
        configs.update(family)
        families[f"k{budget}"] = list(family)
    windows = {}
    for lo, hi in WINDOWS:
        name = f"w{lo}_{hi}"
        windows[name] = spaced(lo, hi, 25, spaces["t"])
    configs.update(windows)
    families["window"] = list(windows)
    return configs, families


class StepSamples:
    """Sample aligned spatial coordinates for correlation and single-step AP."""
    def __init__(self, label, brain, index):
        self.truth = label
        rng = np.random.default_rng(4813 + index)
        self.coords = rng.choice(label.size, 32768, replace=False)
        self.brain_samples = rng.choice(np.flatnonzero(brain), 8192, replace=False)
        self.values = np.empty((len(BASE), len(self.coords)), np.float32)
        self.log_values = np.empty((len(BASE), 4, len(self.brain_samples)), np.float32)

    def __call__(self, steps, logs, first, last):
        for coords, values in ((self.coords, self.values), (self.brain_samples, self.log_values)):
            x, y, z = np.unravel_index(coords, SHAPE)
            use = (z >= first) & (z < last)
            sampled = logs[z[use] - first, :, :, x[use], y[use]].numpy()
            if values.ndim == 2:
                values[:, use] = sampled.max(axis=-1).T
            else:
                values[:, :, use] = sampled.transpose(1, 2, 0)


class NewStepCache:
    """Store new steps like the legacy grouplog cache, each in a singleton group."""
    def __init__(self, split, subject, brain, truth, steps):
        self.path = STEP_CACHE / split / f"{subject}.npz"
        self.brain = brain
        self.truth = truth
        self.steps = tuple(steps)
        self.logs = np.empty((len(self.steps), 4, int(brain.sum())), np.float32)
        self.counts = np.array([truth.sum(), (brain & ~truth).sum(), (~brain).sum()])
        self.sums = np.zeros((len(self.steps), 4, 3), dtype=np.float64)
        # Same axial-major foreground voxel order as the 75–199 cache.
        self.z, self.x, self.y = np.nonzero(brain.transpose(2, 0, 1))

    def __call__(self, steps, logs, first, last):
        use = (self.z >= first) & (self.z < last)
        brain = self.brain[:, :, first:last].transpose(2, 0, 1)
        truth = self.truth[:, :, first:last].transpose(2, 0, 1)
        a = logs.numpy()
        for i, step in enumerate(self.steps):
            residual = a[:, steps.index(step)]
            self.logs[i][:, use] = residual[self.z[use] - first, :, self.x[use], self.y[use]].T
            for group, mask in enumerate((truth, brain & ~truth, ~brain)):
                self.sums[i, :, group] += np.where(
                    mask[:, None], np.maximum(residual, np.log(1e-30)), 0
                ).sum(axis=(0, 2, 3), dtype=np.float64)

    def save(self):
        self.path.parent.mkdir(exist_ok=True, parents=True)
        # -inf represents exact zero residuals. Preserve them through quantization.
        finite = np.isfinite(self.logs)
        offsets = np.sum(self.logs, axis=2, where=finite, dtype=np.float64)
        offsets = (offsets / np.maximum(finite.sum(axis=2), 1)).astype(np.float32)
        self.logs -= offsets[..., None]
        write_npz(
            self.path, compressed=True, steps=np.array(self.steps), brain=np.packbits(self.brain),
            label=np.packbits(self.truth), glog=self.logs.astype(np.float16), goff=offsets,
            tcurve=(self.sums / np.maximum(self.counts, 1)).astype(np.float32),
            counts=self.counts,
        )


def score_volume(scores, truth, low, high):
    scores -= low
    scores /= high - low
    pred = binary_dilation(scores > threshold_yen(scores), structure=DILATION)
    yen = 2 * (pred & truth).sum() / (pred.sum() + truth.sum())
    return yen, threshold_dice(scores, truth, CEILING_THRESHOLDS)


def score_config(directory, ids, split, name, kernel, executor, labels):
    def read(subject):
        if name == "baseline":
            scores, _ = archived(split, subject)
        else:
            with np.load(directory / "maps" / f"{subject}.npz", allow_pickle=False) as saved:
                scores = saved[name]
        return median_filter(scores, size=kernel) if kernel else scores

    scores = np.empty((len(ids), *SHAPE), dtype=np.float32)
    for row, filtered in enumerate(executor.map(read, ids)):
        scores[row] = filtered
    low, high = float(scores.min()), float(scores.max())
    fn = partial(score_volume, low=low, high=high)
    yen, ceiling = zip(*executor.map(fn, scores, labels))
    ap = pooled_average_precision(torch.from_numpy(scores), torch.from_numpy(labels))
    result = dict(auprc=ap, dice=float(np.mean(yen)), ceiling_dice=float(np.mean(ceiling, axis=0).max()),
                  subject_dice=list(yen))
    print(f"{split} MF{kernel} {name}: AP={ap:.6f} Dice={result['dice']:.6f}", flush=True)
    return result


def infer_split(conf, inference, split, stage, subset, configs, panel=()):
    directory = OUTPUT / stage
    paths = directory / "maps"
    paths.mkdir(parents=True, exist_ok=True)
    design_path = directory / "design.json"
    design = {"configs": {name: list(ts) for name, ts in configs.items()},
              "subjects": list(subset), "timing_panel": list(panel)}
    if design_path.exists() and json.loads(design_path.read_text()) != design:
        raise ValueError(f"Cannot resume a different design in {directory}")
    write_json(design_path, design)
    loader = MRI_Volume(conf)
    ids = subject_ids(conf["path_to_csv"])
    steps = {step for ts in configs.values() for step in ts}
    new_steps = sorted(steps - set(BASE))
    step_dir = STEP_CACHE / split
    if new_steps:
        step_dir.mkdir(parents=True, exist_ok=True)
        write_json(step_dir / "manifest.json", {
            "checkpoint_sha256": json.loads((ARCHIVE[split] / "manifest.json").read_text())["checkpoint_sha256"],
            "torch": str(torch.__version__), "device": "mps", "seed": conf["seed"], "batch": BATCH,
            "subjects": list(subset), "steps": new_steps,
            "rng": "legacy for 75–199; independent MPS seed by split-row, timestep and slice batch elsewhere",
            "groups": "singleton per step; brain-only log residuals centered by goff (float32), stored in glog (float16)",
            "voxel_order": "S,H,W over packed H,W,S brain mask",
            "tcurve": "mean log residuals floored at 1e-30, for lesion, healthy-in-brain, and background",
        })
    for index, subject in enumerate(ids):
        if subject not in subset:
            continue
        path = paths / f"{subject}.npz"
        time_path = directory / f"{subject}.json"
        if path.exists() and time_path.exists() and (
            not new_steps or (step_dir / f"{subject}.npz").exists()
        ):
            continue
        start = perf_counter()
        volume, label = loader.dataset[index]
        truth = label.numpy()
        brain = volume.gt(0).any(dim=0).numpy()
        if stage != "calibration" and split == "val":
            with np.load(Path("results/cache/val") / f"{subject}.npz") as prior:
                brain = np.unpackbits(prior["brain"]).reshape(SHAPE).astype(bool)
                if not np.array_equal(np.packbits(truth), prior["label"]):
                    raise ValueError(f"Archived mask mismatch for {subject}")
        batches = inference.batches(volume.unsqueeze(0))
        torch.mps.synchronize()
        prep_seconds = perf_counter() - start
        states = inference.noise_states(batches, index)
        noise_setup_seconds = perf_counter() - start - prep_seconds
        new = NewStepCache(split, subject, brain, truth, new_steps) if new_steps else None
        samples = StepSamples(truth, brain, index) if stage == "calibration" else None
        observer = new or samples
        maps, seconds = inference.predict(batches, states, index, configs, observer)
        if samples:
            maps = {"scores": samples.values, "truth": truth.ravel()[samples.coords],
                    "logs": samples.log_values}
        if new:
            new.save()
        benchmarks = {}
        if subject in panel:
            methods = [("baseline", BASE), *configs.items()]
            # Rotate order across panel subjects to reduce thermal/order bias.
            for method in np.roll(np.arange(len(methods)), panel.index(subject)):
                name, ts = methods[method]
                direct, inference_seconds = inference.predict(batches, states, index, {name: ts})
                reference = archived(split, subject)[0] if name == "baseline" else maps[name]
                np.testing.assert_allclose(direct[name], reference, rtol=1e-5, atol=1e-10)
                benchmarks[name] = inference_seconds
        write_npz(path, **maps)
        write_json(time_path, dict(subject=subject, inference_s=seconds, preprocessing_s=prep_seconds,
                                  noise_replay_setup_s=noise_setup_seconds, timings_s=benchmarks))
        print(f"{stage} {subject} ({index + 1}/{len(ids)}): {seconds:.2f} s", flush=True)


def calibrate(conf, inference):
    ids = subject_ids(conf["path_to_csv"])
    subset = ids[:CALIBRATION_SIZE]
    infer_split(conf, inference, "val", "calibration", subset, {"baseline": BASE})
    directory = OUTPUT / "calibration"
    log_samples, scores, truth = [], [], []
    for subject in subset:
        with np.load(directory / "maps" / f"{subject}.npz") as data:
            log_samples.append(data["logs"].reshape(len(BASE), -1))
            scores.append(data["scores"])
            truth.append(data["truth"])
    log_samples = np.maximum(np.concatenate(log_samples, axis=1), np.log(1e-30))
    correlation = np.corrcoef(log_samples)
    scores = np.concatenate(scores, axis=1)
    aps = [float(average_precision_score(np.concatenate(truth), row)) for row in scores]
    write_json(directory / "diagnostics.json", dict(
        timesteps=BASE, correlation=correlation.tolist(), sampled_single_step_ap=aps,
        subject_ids=subset, noise_fraction=coordinates(inference)["noise"].tolist(),
    ))
    return correlation


def validate(conf, inference):
    diagnostics = OUTPUT / "calibration" / "diagnostics.json"
    correlation = np.asarray(json.loads(diagnostics.read_text())["correlation"])
    configs, groups = candidates(inference, correlation)
    ids = subject_ids(conf["path_to_csv"])[CALIBRATION_SIZE:]
    infer_split(conf, inference, "val", "validation", ids, configs)
    directory = OUTPUT / "validation"
    labels = np.stack([archived("val", subject)[1] for subject in ids])
    results_dir = directory / "metrics"
    results_dir.mkdir(exist_ok=True)
    scores = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for name in ("baseline", *configs):
            path = results_dir / f"{name}.json"
            if not path.exists():
                write_json(path, score_config(directory, ids, "val", name, 5, executor, labels))
            scores[name] = json.loads(path.read_text())
    winners = {
        family: max(names, key=lambda name: (scores[name]["auprc"], scores[name]["dice"]))
        for family, names in groups.items()
    }
    # Freeze the winning window and fixed-window configurations before loading any test maps.
    unique = {}
    for name in winners.values():
        if configs[name] not in unique.values():
            unique[name] = configs[name]
    frozen = dict(selection="Maximum validation MF5 AUPRC, then Yen+dilation mean Dice",
                  validation_subjects=ids, winners=winners, configs=unique, baseline=BASE)
    path = OUTPUT / "frozen.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(frozen)):
        raise ValueError("Existing freeze disagrees with this validation result")
    write_json(path, frozen)
    print(f"Frozen validation choices: {winners}", flush=True)


def test():
    frozen = json.loads((OUTPUT / "frozen.json").read_text())
    conf = setup("test")
    ids = subject_ids(conf["path_to_csv"])
    if len(ids) != 251 or set(ids) & set(frozen["validation_subjects"]):
        raise ValueError("Expected 251 independent BraTS21 test subjects")
    inference = Inference(conf)
    panel = [ids[i] for i in np.linspace(0, len(ids) - 1, TIMING_PANEL_SIZE, dtype=int)]
    configs = frozen["configs"]
    infer_split(conf, inference, "test", "test", ids, configs, panel)
    directory = OUTPUT / "test"
    times = [json.loads((directory / f"{subject}.json").read_text()) for subject in panel]
    labels = np.stack([archived("test", subject)[1] for subject in ids])
    results = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for name in ("baseline", *configs):
            results[name] = {}
            for kernel in FILTERS:
                path = directory / f"{name}_mf{kernel}.json"
                if not path.exists():
                    write_json(path, score_config(directory, ids, "test", name, kernel, executor, labels))
                results[name][kernel] = json.loads(path.read_text())
                result = results[name][kernel]
                baseline = results["baseline"][kernel]
                result["delta_auprc"] = result["auprc"] - baseline["auprc"]
                result["delta_dice"] = result["dice"] - baseline["dice"]
                paired = np.subtract(result["subject_dice"], baseline["subject_dice"])
                resampled = np.random.default_rng(73).choice(paired, size=(10000, len(ids)), replace=True)
                result["paired_dice_delta_ci95"] = np.quantile(resampled.mean(axis=1), [0.025, 0.975]).tolist()
            sec = np.array([row["timings_s"][name] for row in times])
            total = sec + [row["preprocessing_s"] + row["noise_replay_setup_s"] for row in times]
            results[name]["timing"] = dict(
                median_inference_s=float(np.median(sec)), median_total_s=float(np.median(total)),
                inference_iqr_s=np.quantile(sec, [0.25, 0.75]).tolist(),
            )
            for scope in ("inference", "total"):
                results[name]["timing"][f"{scope}_speedup"] = (
                    results["baseline"]["timing"][f"median_{scope}_s"]
                    / results[name]["timing"][f"median_{scope}_s"]
                )
    write_json(OUTPUT / "results.json", dict(subjects=len(ids), timing_subjects=panel, results=results))


def setup(split):
    conf = yaml.safe_load(Path("conf/eval.yml").read_text())
    conf.update(path_to_csv=f"splits/BraTS21/scans_{split}.csv", workers=0)
    with open(conf["model"], "rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    manifest = json.loads((ARCHIVE[split] / "manifest.json").read_text())
    ids = subject_ids(conf["path_to_csv"])
    if (
        manifest["checkpoint_sha256"] != digest
        or (manifest["batch"], manifest["seed"], manifest["start"], manifest["stop"],
            manifest["pyramid"], manifest["torch"]) != (BATCH, conf["seed"], 75, 200, False, torch.__version__)
        or {file.stem for file in ARCHIVE[split].glob("*.npz")} != set(ids)
        or len(ids) != manifest["subjects"]
    ):
        raise ValueError("Archive provenance differs from the requested protocol")
    return conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("verify", "pilot", "calibrate", "validate", "test", "run"))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.stage != "test":
        conf = setup("val")
        inference = Inference(conf)
        if args.stage == "verify" or (args.stage == "run" and not (OUTPUT / "replay_verified").exists()):
            verify(conf, inference)
            (OUTPUT / "replay_verified").touch()
        if args.stage == "pilot":
            timing_pilot(conf, inference)
        if args.stage in ("calibrate", "run"):
            calibrate(conf, inference)
        if args.stage in ("validate", "run"):
            validate(conf, inference)
    if args.stage in ("test", "run"):
        test()
