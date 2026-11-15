"""Produce the BraTS21 E1 tables and figures from the completed run."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from experiments.e1 import ARCHIVE, BASE, BUDGETS, OUTPUT, RANDOM_SEEDS, SHAPE, archived, saved_map


def load(path):
    return json.loads(path.read_text())


def write_diagnostics():
    diagnostics = load(OUTPUT / "calibration" / "diagnostics.json")
    steps = diagnostics["timesteps"]
    quantiles = diagnostics.get("log_residual_quantiles")
    fig, plots = plt.subplots(1, 4 if quantiles else 3,
                              figsize=(17 if quantiles else 13, 3.4), layout="constrained")
    ax = plots[0]
    ax.plot(range(50, 250), np.asarray(diagnostics["noise_fraction"])[50:250] * 100)
    ax.axvspan(75, 199, alpha=0.15, label="Baseline window")
    ax.legend(frameon=False, loc="upper left")
    ax.set(xlabel="Timestep", ylabel="Squared noise coefficient (%)")
    ax = plots[1]
    ax.plot(steps, diagnostics["sampled_single_step_ap"])
    ax.set(xlabel="Timestep", ylabel="Sampled voxel AP", title="Validation diagnostic using lesion labels")
    ax = plots[2]
    correlation = np.asarray(diagnostics["correlation"])
    off_diagonal = np.ma.array(correlation, mask=np.eye(len(steps), dtype=bool))
    heat = ax.imshow(off_diagonal, extent=(74.5, 199.5, 199.5, 74.5), cmap="viridis")
    ax.set(xlabel="Timestep", ylabel="Timestep", title="Foreground correlation across timesteps")
    fig.colorbar(heat, ax=ax, shrink=0.8)
    if quantiles:
        ax = plots[3]
        low, median, high = quantiles
        ax.fill_between(steps, low, high, alpha=0.18, label="10th–90th percentile")
        ax.plot(steps, median, label="Median")
        ax.set(xlabel="Timestep", ylabel="Log squared residual", title="Sampled foreground residuals")
        ax.legend(frameon=False)
    fig.savefig(OUTPUT / "diagnostics.png", dpi=180)
    plt.close(fig)


def write_pareto(results, configs):
    fig, plots = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, metric, label in zip(plots, ("auprc", "dice"), ("Pooled AUPRC", "Mean patient Dice")):
        for name, row in results.items():
            timing, scores = row["timing"], row["5"]
            speedup = timing["inference_speedup"]
            ax.scatter(speedup, scores[metric])
            ax.annotate(f"{len(configs[name])} · {name}", (speedup, scores[metric]),
                        xytext=(4, 5), textcoords="offset points", fontsize=7)
        ax.margins(x=0.18, y=0.2)
        ax.set(xlabel="Measured median inference speedup", ylabel=label)
    fig.savefig(OUTPUT / "pareto_mf5.png", dpi=180)
    plt.close(fig)


def write_examples(results, configs, embedded_baseline=False):
    # Use the first subject in the predeclared timing panel.
    subject = results["timing_subjects"][0]
    if embedded_baseline:
        baseline = saved_map(OUTPUT / "test", subject, "baseline")
        truth = np.unpackbits(saved_map(OUTPUT / "test", subject, "label")).reshape(SHAPE).astype(bool)
    else:
        baseline, truth = archived("test", subject)
    section = truth.sum(axis=(0, 1)).argmax()
    fig, plots = plt.subplots(1, len(configs) + 1, figsize=(2.4 * (len(configs) + 1), 2.8),
                              layout="constrained")
    vmax = np.percentile(baseline[:, :, section], 99.8)
    with np.load(OUTPUT / "test" / "maps" / f"{subject}.npz") as maps:
        for ax, (name, score) in zip(plots, [("125-step baseline", baseline), *((n, maps[n]) for n in configs)]):
            ax.imshow(score[:, :, section].T, origin="lower", cmap="magma", vmin=0, vmax=vmax)
            ax.contour(truth[:, :, section].T, [0.5], colors=["cyan"], linewidths=0.6)
            ax.set_title(name, fontsize=9)
            ax.axis("off")
    fig.suptitle(f"{subject}, axial slice {section} with reference lesion outline", fontsize=10)
    fig.savefig(OUTPUT / "example.png", dpi=180)
    plt.close(fig)


def main(embedded_baseline=False):
    frozen = load(OUTPUT / "frozen.json")
    data = load(OUTPUT / "results.json")
    configs = {"baseline": BASE, **frozen["configs"]}
    manifest = load(OUTPUT / "provenance.json" if embedded_baseline else ARCHIVE["val"] / "manifest.json")
    hardware = manifest["hardware"] if embedded_baseline else "Apple M5 MPS"
    backend = "CUDA" if embedded_baseline else "MPS"
    lines = [
        "# E1 BraTS21 timestep subset selection", "",
        f"{hardware}, PyTorch {manifest['torch']}, frozen `pyramid/232_ema_ckpt.pt`.",
        f"Checkpoint SHA-256 `{manifest['checkpoint_sha256']}`.", "",
        f"Calibration used eight validation subjects for label-free residual correlations. "
        f"Selection used the remaining {len(frozen['validation_subjects'])} validation subjects, "
        f"maximizing MF5 pooled AUPRC and breaking ties by MF5 Yen+dilation Dice. "
        f"The winners were frozen in `frozen.json` before scoring {data['subjects']} test subjects.", "",
        "## Validation comparisons", "",
        "| Candidate | Steps | MF5 AUPRC | MF5 Dice |",
        "|---|---:|---:|---:|",
    ]
    validation = OUTPUT / "validation"
    design = load(validation / "design.json")["configs"]
    validation_metrics = {name: load(validation / "metrics" / f"{name}.json")
                          for name in dict.fromkeys(("baseline", *design))}
    for name, row in validation_metrics.items():
        lines.append(f"| {name} | {len(design.get(name, BASE))} | {row['auprc']:.6f} | {row['dice']:.6f} |")
    lines += ["", "| Random budget | AUPRC mean ± SD | Dice mean ± SD |", "|---:|---:|---:|"]
    for budget in BUDGETS:
        rows = [validation_metrics[f"k{budget}_random{seed}"] for seed in RANDOM_SEEDS]
        ap, dice = ([row[key] for row in rows] for key in ("auprc", "dice"))
        lines.append(f"| {budget} | {np.mean(ap):.6f} ± {np.std(ap, ddof=1):.6f} | "
                     f"{np.mean(dice):.6f} ± {np.std(dice, ddof=1):.6f} |")
    lines += ["", "## Frozen test results", "",
              "| Config | Steps | Median filter | AUPRC | Δ AUPRC | Yen + dilation Dice | Δ Dice [95% CI] | Ceiling Dice |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in data["results"].items():
        for kernel in ("0", "3", "5"):
            score = row[kernel]
            lo, hi = score["paired_dice_delta_ci95"]
            lines.append(f"| {name} | {len(configs[name])} | {kernel} | {score['auprc']:.6f} | "
                         f"{score['delta_auprc']:+.6f} | {score['dice']:.6f} | "
                         f"{score['delta_dice']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {score['ceiling_dice']:.6f} |")
    lines += ["", f"## Synchronized {backend} timing", "",
              f"Each method ran separately on the same {len(data['timing_subjects'])} predetermined test subjects. "
              f"Inference timing includes all slices, residual transfers and score aggregation. {backend} synchronizes "
              "before and after every timed run. Total latency adds the same subject's preprocessing and noise "
              "state preparation. Both exclude offline median filtering, metric computation and cache I/O.", "",
              "| Config | Inference median [IQR] (s) | Inference speedup | Total median (s) | Total speedup |",
              "|---|---:|---:|---:|---:|"]
    for name, row in data["results"].items():
        timing = row["timing"]
        lo, hi = timing["inference_iqr_s"]
        lines.append(f"| {name} | {timing['median_inference_s']:.2f} [{lo:.2f}, {hi:.2f}] | "
                     f"{timing['inference_speedup']:.2f}× | {timing['median_total_s']:.2f} | "
                     f"{timing['total_speedup']:.2f}× |")
    lines += [
        "", "The random variability table varies subset-selection seeds with the evaluation noise fixed. "
        "The main comparison uses one matched evaluation-noise realization per subject.",
        "", "The new-step caches store individual timestep log residuals in the same "
        "centered float16 foreground format as the grouped archive, with packed brain/label masks and per-region "
        "statistics. Full-volume inference maps were retained separately for exact filtering and scoring.",
    ]
    if embedded_baseline and all("median_peak_cuda_gib" in row["timing"] for row in data["results"].values()):
        lines += ["", "Peak PyTorch CUDA allocation, median across timing subjects", "",
                  "| Config | GiB |", "|---|---:|"]
        for name, row in data["results"].items():
            lines.append(f"| {name} | {row['timing']['median_peak_cuda_gib']:.3f} |")
    if (OUTPUT / "noise_seeds.json").exists():
        seeds = load(OUTPUT / "noise_seeds.json")
        lines += ["", "## Evaluation noise sensitivity", "",
                  f"The baseline and selected 25-step schedule were reevaluated with two independent noise seeds "
                  f"on {len(seeds['subjects'])} calibration subjects. These scores are descriptive.", "",
                  "| Seed | Config | MF5 AUPRC | MF5 Dice |", "|---:|---|---:|---:|"]
        for seed, methods in seeds["trials"].items():
            for name, row in methods.items():
                lines.append(f"| {seed} | {name} | {row['auprc']:.6f} | {row['dice']:.6f} |")
    if embedded_baseline:
        lines += [
            "", "## Reproduction", "",
            f"Branch commit `{manifest['identity']['commit']}`. The CUDA run used a freshly evaluated "
            "125-step CUDA baseline with noise matched within this experiment. See `kaggle/README.md` "
            "at this commit for the staged CLI commands.",
        ]
    else:
        lines += ["", "## Commands", "", "```bash",
        "git switch -c codex/brats21-e1 codex/brats21-mps-reproduction",
        "env PYTORCH_ENABLE_MPS_FALLBACK=0 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 caffeinate -i .venv/bin/python -u -m experiments.e1 verify",
        "env PYTORCH_ENABLE_MPS_FALLBACK=0 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 caffeinate -i .venv/bin/python -u -m experiments.e1 run",
        "env OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.e1_report",
        "```"]
    lines += ["", "- [Noise schedule, single-step AP and correlations](diagnostics.png)",
        "- [Accuracy versus measured speedup](pareto_mf5.png)", "- [Example maps](example.png)", "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines))
    write_diagnostics()
    write_pareto(data["results"], configs)
    write_examples(data, frozen["configs"], embedded_baseline)


if __name__ == "__main__":
    main()
