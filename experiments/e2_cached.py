"""Score archived BraTS21 maps and evaluate a synthetic-calibrated threshold."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, generate_binary_structure, grey_dilation, median_filter
from skimage.filters import threshold_yen
from tqdm import tqdm

from andi.evaluation.metrics import pooled_average_precision

SHAPE = (128, 128, 155)
KERNELS = (0, 3, 5)
DILATION = generate_binary_structure(3, 1)
RAW_THRESHOLDS = np.geomspace(1e-6, 1e-3, 61).astype(np.float32)
CEILING_THRESHOLDS = np.rint(np.arange(0.01, 0.3, 0.001) * 1000).astype(np.float32) / 1000
PAPER = {0: (0.654, 0.587, 0.388), 3: (0.829, 0.747, 0.701), 5: (0.858, 0.781, 0.766)}


def split_ids(split):
    with Path(f"splits/BraTS21/scans_{split}.csv").open() as stream:
        return [row["BraTS21ID"] for row in csv.DictReader(stream)]


def unpack_mask(packed):
    return np.unpackbits(packed).reshape(SHAPE).astype(bool)


def filter_scores(scores, size):
    return median_filter(scores, size=size) if size else scores


def threshold_dice(scores, truth, thresholds, valid=None):
    positives = truth.sum()
    positive_scores = np.sort(scores[truth])
    all_scores = np.sort(scores.ravel() if valid is None else scores[valid])
    tp = positives - np.searchsorted(positive_scores, thresholds, side="right")
    predicted = all_scores.size - np.searchsorted(all_scores, thresholds, side="right")
    return 2 * tp / (positives + predicted)


def synthetic_curves(path):
    subject, _ = path.stem.rsplit("_", 1)
    with np.load(Path("results/cache/val") / f"{subject}.npz", allow_pickle=False) as clean:
        scores = clean["score_full"]
        brain = unpack_mask(clean["brain"])
    with np.load(path, allow_pickle=False) as synthetic:
        changed = synthetic["z"]
        scores[:, :, changed] = synthetic["score_slices"]
        lesions = unpack_mask(synthetic["mask"])
        valid = (brain | lesions) & ~unpack_mask(synthetic["ignore"])
    lesions &= valid
    if not lesions.any() or not np.isfinite(scores).all():
        raise ValueError(f"Invalid synthetic case: {path}")
    curves = []
    for kernel in KERNELS:
        filtered = filter_scores(scores, kernel)
        # Thresholding a grey dilation gives exactly the binary dilation of the mask.
        dilated = grey_dilation(filtered, footprint=DILATION, mode="constant", cval=-np.inf)
        curves.append([
            threshold_dice(filtered, lesions, RAW_THRESHOLDS, valid),
            threshold_dice(dilated, lesions, RAW_THRESHOLDS, valid),
        ])
    return curves


def calibrate(executor, output):
    cases = sorted(Path("results/cache/synthetic").glob("*.npz"))
    subjects = sorted({case.stem.rsplit("_", 1)[0] for case in cases})
    if set(subjects) != set(split_ids("val")) or set(subjects) & set(split_ids("test")):
        raise ValueError("Synthetic source subjects must match the independent validation split")
    expected = {f"{subject}_{kind}" for subject in subjects for kind in ("hyper", "multi", "patch")}
    if {case.stem for case in cases} != expected:
        raise ValueError("Expected three synthetic lesion families for every source subject")
    held_out = set(np.random.default_rng(73).permutation(subjects)[::5])
    is_holdout = np.array([case.stem.rsplit("_", 1)[0] in held_out for case in cases])
    curves = np.asarray(list(tqdm(
        executor.map(synthetic_curves, cases), total=len(cases), desc="Synthetic calibration"
    )))
    fit_curves = curves[~is_holdout].mean(axis=0)
    best = fit_curves.argmax(axis=-1)
    thresholds = RAW_THRESHOLDS[best]
    held_out_curves = curves[is_holdout].mean(axis=0)
    scores = np.take_along_axis(held_out_curves, best[..., None], axis=-1)[..., 0]
    result = {
        "calibration_subjects": sorted(set(subjects) - held_out),
        "holdout_subjects": sorted(held_out),
        "median_filters": KERNELS,
        "rules": ["no_dilation", "dilation"],
        "threshold_grid": RAW_THRESHOLDS.tolist(),
        "calibration_mean_dice_grid": fit_curves.tolist(),
        "thresholds": thresholds.tolist(),
        "synthetic_holdout_mean_dice": scores.tolist(),
    }
    (output / "e2_calibration.json").write_text(json.dumps(result, indent=2))
    print(f"Frozen raw thresholds (MF0, MF3, MF5; no dilation, dilation): {thresholds}", flush=True)
    return thresholds


def segmentation_metrics(pred, truth):
    tp = np.count_nonzero(pred & truth)
    fp = int(pred.sum()) - tp
    positives = int(truth.sum())
    return (2 * tp / (positives + tp + fp), fp / (truth.size - positives))


def evaluate_subject(scores, truth, lo, hi, thresholds):
    normalized = (scores - lo) / (hi - lo)
    ceiling = threshold_dice(normalized, truth, CEILING_THRESHOLDS)
    yen = normalized > threshold_yen(normalized)
    fixed = scores > thresholds[0]
    fixed_with_dilation = binary_dilation(scores > thresholds[1], structure=DILATION)
    metrics = [
        segmentation_metrics(pred, truth)
        for pred in (yen, binary_dilation(yen, structure=DILATION), fixed, fixed_with_dilation)
    ]
    scores[:] = normalized
    return ceiling, metrics


def read_archived(path):
    with np.load(path, allow_pickle=False) as data:
        scores, truth = data["score"], data["label"]
    if scores.shape != SHAPE or scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise ValueError(f"Invalid archived anomaly map {path}")
    if truth.shape != SHAPE or truth.dtype != bool or not 0 < truth.sum() < truth.size:
        raise ValueError(f"Invalid archived mask {path}")
    return scores, truth


def evaluate_kernel(executor, paths, kernel, thresholds):
    scores = np.empty((len(paths), *SHAPE), dtype=np.float32)
    labels = np.empty(scores.shape, dtype=bool)

    def load_map(i):
        raw, labels[i] = read_archived(paths[i])
        scores[i] = filter_scores(raw, kernel)

    for _ in tqdm(executor.map(load_map, range(len(paths))), total=len(paths), desc=f"MF{kernel} maps"):
        pass
    lo, hi = scores.min(), scores.max()

    def score_subject(i):
        return evaluate_subject(scores[i], labels[i], lo, hi, thresholds)

    evaluations = list(tqdm(
        executor.map(score_subject, range(len(paths))), total=len(paths), desc=f"MF{kernel} Dice"
    ))
    ceiling, metrics = zip(*evaluations)
    dice_grid = np.mean(ceiling, axis=0)
    metrics = np.asarray(metrics)
    print(f"MF{kernel} pooled AP", flush=True)
    ap = pooled_average_precision(torch.from_numpy(scores), torch.from_numpy(labels))
    sizes = labels.sum(axis=(1, 2, 3))
    thirds = np.array_split(np.argsort(sizes, kind="stable"), 3)
    deltas = metrics[:, 3, 0] - metrics[:, 1, 0]
    samples = np.random.default_rng(73).choice(deltas, (10000, len(deltas)), replace=True).mean(axis=1)
    result = {
        "subjects": len(paths),
        "paper": dict(zip(("auprc", "ceiling_dice", "yen_dice"), PAPER[kernel])),
        "auprc": ap,
        "ceiling_dice": float(dice_grid.max()),
        "ceiling_grid": list(zip(CEILING_THRESHOLDS.astype(float).tolist(), dice_grid.tolist())),
        "rules": {
            name: {"dice": float(dice), "fpr": float(fpr)}
            for name, (dice, fpr) in zip(
                ("yen", "yen_dilation", "synthetic", "synthetic_dilation"), metrics.mean(axis=0)
            )
        },
        "paired_dice_delta_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "lesion_size_thirds": [
            {
                "subjects": len(group),
                "lesion_voxels_range": [int(sizes[group].min()), int(sizes[group].max())],
                "yen_dice": float(metrics[group, 1, 0].mean()),
                "synthetic_dice": float(metrics[group, 3, 0].mean()),
            }
            for group in thirds
        ],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/brats21-mps/saved-baseline"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    maps = args.output / "maps"
    with Path("pyramid/232_ema_ckpt.pt").open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    for manifest in (maps / "manifest.json", Path("results/cache/val/manifest.json")):
        metadata = json.loads(manifest.read_text())
        if metadata["checkpoint_sha256"] != digest or (metadata["start"], metadata["stop"]) != (75, 200):
            raise ValueError(f"Cache does not match the frozen checkpoint and baseline: {manifest}")
    ids = split_ids("test")
    if len(ids) != 251 or {path.stem for path in maps.glob("*.npz")} != set(ids):
        raise ValueError("Baseline cache does not match all 251 test subjects")
    paths = [maps / f"{subject}.npz" for subject in ids]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        thresholds = calibrate(executor, args.output)
        for kernel, selected in zip(KERNELS, thresholds):
            result = evaluate_kernel(executor, paths, kernel, selected)
            (args.output / f"metrics_mf{kernel}.json").write_text(json.dumps(result, indent=2))
            summary = {key: value for key, value in result.items() if key != "ceiling_grid"}
            print(f"MF{kernel}: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
