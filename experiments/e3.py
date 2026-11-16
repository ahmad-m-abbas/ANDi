"""Fit and evaluate timestep/modality residual calibration with a frozen ANDi model."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, median_filter
from scipy.stats import spearmanr
from skimage.filters import threshold_yen
import torch
import yaml

from experiments import e1
from experiments.e1_cuda import CHECKPOINT_SHA, CudaInference, digest, gpu_provenance, read as load_json

REF_CASES = 16
DIAGNOSTIC_CASES = 8
VOXELS_PER_CASE = 2048
DELTA = 1e-12
SCALE_FLOOR = 0.1
LESION_MARGIN = 3
METHODS = ("baseline", "mean_clip", "median_clip", "mean_softplus", "median_softplus")
MODALITIES = ("FLAIR", "T1", "T1ce", "T2")
GRID = (8, 8, 8)


def subjects(path):
    with path.open() as stream:
        return [next(iter(row.values())) for row in csv.DictReader(stream)]


def make_design(smoke, transfer):
    val = subjects(Path("splits/BraTS21/scans_val.csv"))
    test = subjects(Path("splits/BraTS21/scans_test.csv"))
    if len(val) != 62 or len(test) != 251 or len(set(val + test)) != 313:
        raise ValueError("Expected independent BraTS21 validation and test splits")
    reference_size = 2 if smoke else REF_CASES
    diagnostic_size = 2 if smoke else DIAGNOSTIC_CASES
    design = {
        "smoke": smoke,
        "reference": val[:reference_size],
        "validation": val[reference_size:reference_size + diagnostic_size],
        "test": [] if smoke else test,
        "ljubljana": subjects(Path("splits/Shifts_MS/ljubljana.csv")) if transfer and not smoke else [],
        "timesteps": list(e1.BASE),
        "modalities": list(MODALITIES),
        "methods": list(METHODS),
        "delta": DELTA,
        "scale_floor": SCALE_FLOOR,
        "sample_limit_per_subject": VOXELS_PER_CASE,
        "lesion_margin_voxels": LESION_MARGIN,
        "reference_type": "nonlesion foreground voxels in BraTS21 tumor-bearing validation scans",
    }
    return design


def sampled_voxels(brain, lesion, row):
    eligible = brain & ~binary_dilation(lesion, structure=e1.DILATION, iterations=LESION_MARGIN)
    voxels = np.flatnonzero(eligible)
    if not len(voxels):
        raise ValueError("No nonlesion foreground voxels after excluding the lesion margin")
    rng = np.random.default_rng(731_003 + row)
    return np.column_stack(np.unravel_index(
        rng.choice(voxels, min(len(voxels), VOXELS_PER_CASE), replace=False), brain.shape,
    ))


class ResidualObserver:
    def __init__(self, coordinates, reference=None):
        self.coordinates = coordinates
        self.samples = np.empty((len(coordinates), len(e1.BASE), len(MODALITIES)), np.float32)
        self.reference = reference
        self.maps = {} if reference is None else {
            method: np.empty(e1.SHAPE, np.float32) for method in METHODS[1:]
        }

    def __call__(self, steps, logs, first, last):
        if tuple(steps) != e1.BASE:
            raise ValueError("E3 requires the complete baseline timestep window")
        values = logs.numpy()
        x, y, z = self.coordinates.T
        within = (z >= first) & (z < last)
        log_delta = np.float32(np.log(DELTA))
        self.samples[within] = np.logaddexp(values[z[within] - first, :, :, x[within], y[within]], log_delta)
        if self.reference is None:
            return

        stable = np.logaddexp(values, log_delta)
        for family in ("mean", "median"):
            center, scale = self.reference[family]
            zscore = (stable - center[None, :, :, None, None]) / scale[None, :, :, None, None]
            self.maps[f"{family}_clip"][:, :, first:last] = (
                np.maximum(zscore, 0).mean(axis=1).max(axis=1).transpose(1, 2, 0)
            )
            np.logaddexp(np.float32(0), zscore, out=zscore)
            self.maps[f"{family}_softplus"][:, :, first:last] = (
                zscore.mean(axis=1).max(axis=1).transpose(1, 2, 0)
            )


def worker(args):
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid E3 shard index")
    root = args.work
    config = load_json(root / "design.json")
    design = config["design"]
    if digest(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Unexpected checkpoint hash")
    backend = gpu_provenance()
    stamp = root / f"backend-{args.phase}-{args.shard}.json"
    if stamp.exists() and load_json(stamp) != backend:
        raise ValueError("Cannot resume maps from a different GPU environment")
    e1.write_json(stamp, backend)

    split = "test" if args.phase in ("test", "ljubljana") else "val"
    conf = yaml.safe_load(Path("conf/eval.yml").read_text())
    csv_path = (Path("splits/Shifts_MS/ljubljana.csv") if args.phase == "ljubljana"
                else Path(f"splits/BraTS21/scans_{split}.csv"))
    conf.update(path_to_csv=str(csv_path), dataset_path=str(args.data), model=str(args.checkpoint), workers=0)
    dataset = e1.MRI_Volume(conf, shift=args.phase == "ljubljana").dataset
    inference = CudaInference(conf, split)
    # Keep Ljubljana draws distinct from BraTS21 test draws.
    if args.phase == "ljubljana":
        inference.seed += 1000
    reference = None
    if args.phase != "reference":
        path = root / "reference.json"
        stats = load_json(path)
        if stats["identity"] != config["identity"] or stats["subjects"] != design["reference"]:
            raise ValueError("Reference statistics do not match the frozen cohort and checkpoint")
        reference = {
            name: tuple(np.asarray(stats[name][key], dtype=np.float32) for key in ("center", "scale"))
            for name in ("mean", "median")
        }
        freeze = root / f"freeze-{args.phase}-{args.shard}.json"
        reference_sha256 = digest(path)
        if freeze.exists() and load_json(freeze)["reference_sha256"] != reference_sha256:
            raise ValueError("Cannot resume maps generated with a different reference")
        e1.write_json(freeze, {"reference_sha256": reference_sha256})
    ids = design[args.phase]
    directory = root / args.phase / "maps"
    directory.mkdir(parents=True, exist_ok=True)
    lookup = {name: i for i, name in enumerate(subjects(csv_path))}
    for subject in ids[args.shard::args.shards]:
        output = directory / f"{subject}.npz"
        if output.exists():
            continue
        index = lookup[subject]
        volume, truth = dataset[index]
        if tuple(truth.shape) != e1.SHAPE:
            raise ValueError(f"Unexpected shape for {subject}")
        lesion = truth.numpy()
        brain = (volume > 0).any(dim=0).numpy()
        coordinates = sampled_voxels(brain, lesion, index) if args.phase in ("reference", "validation") else np.empty((0, 3), int)
        observer = ResidualObserver(coordinates, reference)
        batches = inference.batches(volume.unsqueeze(0))
        maps, elapsed = inference.predict(batches, None, index, {"baseline": e1.BASE}, observer)
        arrays = {}
        if reference is not None:
            arrays.update(maps, **observer.maps, brain=np.packbits(brain), label=np.packbits(lesion))
        if len(coordinates):
            arrays.update(samples=observer.samples, coordinates=coordinates)
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f"Nonfinite residual or anomaly score for {subject}")
        e1.write_npz(output, compressed=True, **arrays)
        print(f"E3 {args.phase} {subject}: {elapsed:.2f} s", flush=True)


def files_for(root, phase, ids):
    directory = root / phase / "maps"
    if {path.stem for path in directory.glob("*.npz")} != set(ids):
        raise ValueError(f"Saved {phase} subjects differ from the planned cohort")
    return [directory / f"{subject}.npz" for subject in ids]


def fit(root, config):
    ids = config["design"]["reference"]
    values, coordinates, counts = [], [], []
    for path in files_for(root, "reference", ids):
        with np.load(path, allow_pickle=False) as saved:
            values.append(saved["samples"])
            coordinates.append(saved["coordinates"])
            counts.append(len(values[-1]))
    sample = np.concatenate(values, axis=0)
    mean = sample.mean(axis=0, dtype=np.float64)
    std = sample.std(axis=0, dtype=np.float64)
    median = np.median(sample, axis=0)
    mad = 1.4826 * np.median(np.abs(sample - median), axis=0)
    # Summarize spatial variation without using it in any calibration method.
    coarse_grid = np.concatenate(coordinates, axis=0) * np.asarray(GRID) // np.asarray(e1.SHAPE)
    cells = np.ravel_multi_index(coarse_grid.T, GRID)
    counts_grid = np.bincount(cells, minlength=np.prod(GRID))
    within_cell_variance = np.zeros(np.prod(GRID), np.float64)
    normalized = (sample - mean) / np.maximum(std, SCALE_FLOOR)
    for cell in np.flatnonzero(counts_grid >= 32):
        within_cell_variance[cell] = normalized[cells == cell].var(axis=0, dtype=np.float64).mean()

    output = {
        "identity": config["identity"], "subjects": ids,
        "timesteps": list(e1.BASE), "modalities": list(MODALITIES),
        "delta": DELTA, "scale_floor": SCALE_FLOOR, "samples_per_timestep_modality": len(sample),
        "counts_by_subject": dict(zip(ids, counts)), "coarse_grid": list(GRID),
        "reference_voxels_by_cell": counts_grid.tolist(),
        "mean_standardized_variance_by_cell": within_cell_variance.tolist(),
        "mean": {"center": mean.tolist(), "scale": np.maximum(std, SCALE_FLOOR).tolist(), "std": std.tolist()},
        "median": {"center": median.tolist(), "scale": np.maximum(mad, SCALE_FLOOR).tolist(), "scaled_mad": mad.tolist()},
    }
    path = root / "reference.json"
    if path.exists():
        if load_json(path) != output:
            raise ValueError("Saved reference statistics differ from this fit")
    else:
        e1.write_json(path, output)
    print(f"E3 reference frozen from {len(ids)} subjects and {len(sample)} nonlesion samples", flush=True)
    return output


def score(root, phase, ids):
    directory = root / phase
    paths = files_for(root, phase, ids)
    labels = np.empty((len(ids), *e1.SHAPE), bool)
    normal = np.empty_like(labels)
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as saved:
            labels[index] = np.unpackbits(saved["label"]).reshape(e1.SHAPE)
            brain = np.unpackbits(saved["brain"]).reshape(e1.SHAPE).astype(bool)
            normal[index] = brain & ~labels[index]
            if not normal[index].any():
                raise ValueError(f"No nonlesion evaluation voxels for {path.stem}")
    directory.joinpath("metrics").mkdir(exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for method in METHODS:
            rows[method] = {}
            for mf in e1.FILTERS:
                path = directory / "metrics" / f"{method}_mf{mf}.json"
                if not path.exists():
                    values = e1.score_config(directory, ids, phase, method, mf, executor, labels,
                                             embedded_baseline=True, normal=normal)
                    e1.write_json(path, values)
                rows[method][str(mf)] = load_json(path)
    return rows


def diagnostics(root, ids, reference, output):
    collected = []
    anatomy_lesions = np.zeros(GRID, dtype=np.int64)
    anatomy_brain = np.zeros_like(anatomy_lesions)
    edges = [(np.arange(bins + 1) * size + bins - 1) // bins for size, bins in zip(e1.SHAPE, GRID)]
    for file in files_for(root, "validation", ids):
        with np.load(file, allow_pickle=False) as saved:
            collected.append(saved["samples"])
            brain = np.unpackbits(saved["brain"]).reshape(e1.SHAPE).astype(bool)
            lesion = np.unpackbits(saved["label"]).reshape(e1.SHAPE).astype(bool)
        for cell in np.ndindex(GRID):
            region = tuple(slice(edge[i], edge[i + 1]) for i, edge in zip(cell, edges))
            anatomy_brain[cell] += brain[region].sum()
            anatomy_lesions[cell] += (lesion[region] & brain[region]).sum()

    raw = np.concatenate(collected)
    z_edges = np.linspace(-12, 12, 97)
    raw_edges = np.linspace(np.log(DELTA), 6, 121)
    arrays = {"timesteps": np.asarray(e1.BASE), "modalities": np.asarray(MODALITIES),
              "raw_edges": raw_edges, "z_edges": z_edges}
    for family, values, bins in [
        ("raw", raw, raw_edges),
        *((family, (raw - np.asarray(reference[family]["center"]))
           / np.asarray(reference[family]["scale"]), z_edges) for family in ("mean", "median")),
    ]:
        hist = np.empty((len(e1.BASE), len(MODALITIES), len(bins) - 1), dtype=np.int64)
        below = (values < bins[0]).sum(axis=0)
        above = (values > bins[-1]).sum(axis=0)
        for step in range(len(e1.BASE)):
            for channel in range(len(MODALITIES)):
                hist[step, channel] = np.histogram(values[:, step, channel], bins=bins)[0]
        arrays.update({f"{family}_counts": hist, f"{family}_below": below, f"{family}_above": above})
    e1.write_npz(output / "residual_histograms.npz", compressed=True, **arrays)
    variance = np.asarray(reference["mean_standardized_variance_by_cell"])
    eligible = (np.asarray(reference["reference_voxels_by_cell"]) >= 32) & (anatomy_brain.ravel() > 0)
    prevalence = anatomy_lesions.ravel()[eligible] / anatomy_brain.ravel()[eligible]
    valid = eligible.sum() >= 3 and np.ptp(prevalence) > 0 and np.ptp(variance[eligible]) > 0
    rho = float(spearmanr(variance[eligible], prevalence).statistic) if valid else None
    e1.write_json(output / "anatomy_diagnostic.json", {
        "subjects": ids, "coarse_grid": list(GRID), "compared_cells": int(eligible.sum()), "spearman_rho": rho,
        "foreground_voxels_by_cell": anatomy_brain.ravel().tolist(),
        "lesion_voxels_by_cell": anatomy_lesions.ravel().tolist(),
    })
    return arrays, rho


def render_figures(root, output, ids, reference, histogram, results, phase):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for family in ("raw", "mean", "median"):
        ax = axes[0 if family == "raw" else 1]
        edges = histogram["raw_edges" if family == "raw" else "z_edges"]
        counts = histogram[f"{family}_counts"].sum(axis=(0, 1))
        ax.stairs(counts / counts.sum(), edges,
                  label={"raw": "Raw log residual", "mean": "Mean / SD", "median": "Median / MAD"}[family])
    for ax, label in zip(axes, ("Log residual", "Standardized residual (z)")):
        ax.set(xlabel=label, ylabel="Fraction within plotted range")
        ax.legend(frameon=False)
    fig.savefig(output / "residual_distributions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 4.5), layout="constrained")
    for col, (family, label) in enumerate((("mean", "Mean / SD"), ("median", "Median / MAD"))):
        for row, field in enumerate(("center", "scale")):
            ax = axes[row, col]
            im = ax.imshow(np.asarray(reference[family][field]).T, aspect="auto",
                           extent=(74.5, 199.5, 3.5, -0.5))
            ax.set(title=f"{label} · {field}", xlabel="Timestep", yticks=range(4), yticklabels=MODALITIES)
            fig.colorbar(im, ax=ax)
    fig.savefig(output / "reference_statistics.png", dpi=160)
    plt.close(fig)

    subject = ids[0]
    methods = ("baseline", "mean_clip", "median_clip")
    fig, axes = plt.subplots(1, len(methods), figsize=(9, 3.4), layout="constrained")
    with np.load(root / phase / "maps" / f"{subject}.npz", allow_pickle=False) as saved:
        truth = np.unpackbits(saved["label"]).reshape(e1.SHAPE).astype(bool)
        brain = np.unpackbits(saved["brain"]).reshape(e1.SHAPE).astype(bool)
        section = int(np.argmax(truth.sum(axis=(0, 1))))
        for ax, method in zip(axes, methods):
            scores = median_filter(saved[method], size=5)
            metrics = results[method]["5"]
            scores = (scores - metrics["score_min"]) / (metrics["score_max"] - metrics["score_min"])
            pred = binary_dilation(scores > threshold_yen(scores), structure=e1.DILATION)
            ax.imshow(scores[:, :, section].T, origin="lower", cmap="gray",
                      vmax=max(float(np.quantile(scores[:, :, section], 0.99)), 1e-6))
            overlay = np.zeros((*truth.shape[:2], 4))
            overlay[(pred & brain & ~truth)[:, :, section]] = [1.0, 0.45, 0.05, 0.7]
            ax.imshow(overlay.transpose(1, 0, 2), origin="lower")
            if truth[:, :, section].any():
                ax.contour(truth[:, :, section].T, [0.5], colors=["cyan"], linewidths=0.7)
            ax.set_title(method.replace("_", " "))
            ax.axis("off")
    fig.suptitle(f"{subject} · orange false positives · cyan lesion outline")
    fig.savefig(output / "false_positive_example.png", dpi=160)
    plt.close(fig)


def report(root, output, config, reference):
    design = config["design"]
    groups = [phase for phase in ("validation", "test", "ljubljana") if design[phase]]
    results = {phase: score(root, phase, design[phase]) for phase in groups}
    histogram, rho = diagnostics(root, design["validation"], reference, output)
    e1.write_json(output / "reference.json", reference)
    e1.write_json(output / "results.json", {"protocol": config, "results": results})
    lines = [
        "# E3 residual reference calibration" + (" · smoke test" if design["smoke"] else ""), "",
        f"Commit `{config['identity']['commit']}`. The 125-step baseline and all four calibrated methods share each noise draw.",
        f"The reference used {len(design['reference'])} BraTS21 validation subjects and "
        f"{reference['samples_per_timestep_modality']:,} sampled nonlesion foreground voxels per timestep and modality. "
        f"The masks excluded lesions and a {LESION_MARGIN}-voxel dilation. The reference is a nonlesion proxy.",
        f"The primary metric is pooled voxel AUPRC with 5×5×5 median filtering. Dice uses Yen thresholding and dilation. "
        f"Nonlesion FPR is the mean per-subject false-positive fraction inside nonlesion foreground.", "",
    ]
    for phase, methods in results.items():
        baseline = methods["baseline"]
        lines += [f"## {phase.capitalize()} · {len(design[phase])} subjects", "",
                  "| Filter | Method | AUPRC | Δ AUPRC | Yen Dice | Δ Dice | Nonlesion FPR |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for mf in e1.FILTERS:
            for method, by_filter in methods.items():
                row, ref = by_filter[str(mf)], baseline[str(mf)]
                lines.append(f"| {mf} | {method} | {row['auprc']:.6f} | {row['auprc']-ref['auprc']:+.6f} | "
                             f"{row['dice']:.6f} | {row['dice']-ref['dice']:+.6f} | {row['nonlesion_fpr']:.6f} |")
        lines += [""]
    if rho is not None:
        lines += [f"The coarse spatial check found Spearman ρ = {rho:.3f} between reference variance and validation lesion "
                  "prevalence in eligible grid cells. This is descriptive and uses BraTS image coordinates.", ""]
    lines += [
        "[Reference statistics](reference_statistics.png) · [Held-out residual distributions](residual_distributions.png) · "
        "[False-positive example](false_positive_example.png)", "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    example_phase = "test" if "test" in results else "validation"
    render_figures(root, output, design[example_phase], reference, histogram, results[example_phase], example_phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reference", "validation", "test", "ljubljana"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    torch.set_num_threads(4)
    worker(parser.parse_args())
