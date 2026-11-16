# Run BraTS21 E1 on Kaggle

The jobs run in Kaggle after `kaggle kernels push` returns. All inference uses the frozen EMA checkpoint, the repository's BraTS21 split and preprocessing, Gaussian evaluation noise, and the E1 postprocessing and metrics. CUDA uses its own matched noise draws and 125-step baseline. Its speedup describes the selected Kaggle GPU.

## Setup

1. Push the branch to GitHub and keep the same checkout for preparing every job.

   ```bash
   git push -u origin codex/stack-05-e3
   ```

2. Use the private [Kaggle Model version](https://www.kaggle.com/models/ahmadabbas176/232-ema-ckpt-pt/PyTorch/default/1) as the checkpoint input. The runner checks the SHA-256 before loading it. The MRI input defaults to [BraTS 2021 Task 1](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1). The launcher accepts the extracted NIfTI folders or a mounted training tar and checks that all five files exist for each assigned subject.

3. Choose one GPU type for all inference jobs. These examples use one GPU from the T4 runtime. Use a different accelerator consistently if available in your account.

Run the following from the repository root in Bash. Use a new kernel name prefix for each independent experiment so the downstream jobs attach the intended outputs.

```bash
OWNER=ahmadabbas176
CHECKPOINT_MODEL="$OWNER/232-ema-ckpt-pt/pytorch/default/1"
PREFIX=andi-e1
ACCELERATOR=NvidiaTeslaT4
JOB_DIR="$(mktemp -d)"

prepare() {
  local stage="$1" name="$2"
  shift 2
  python3 -B kaggle/prepare.py "$stage" \
    --id "$OWNER/$PREFIX-$name" --out "$JOB_DIR/$name" \
    --checkpoint-model "$CHECKPOINT_MODEL" "$@"
}
push_gpu() {
  kaggle kernels push -p "$JOB_DIR/$1" --accelerator "$ACCELERATOR"
}
```

The generated jobs clone the public GitHub branch and check out the exact commit recorded by `prepare.py`.

Use the same source, checkpoint model version, BraTS21 dataset version, and GPU type throughout. GPU stage manifests record the commit, dataset and model references, checkpoint hash, GPU model and CUDA software versions. The join checks matching provenance, subject coverage and the validation freeze.

## One notebook cell

With the same two inputs mounted, Internet enabled, and GPU T4 ×2 selected, put the cell below in a Kaggle notebook and use **Save Version → Save & Run All**. Start with `PROBE = True`. The preflight uses two validation cases to exercise all candidate schedules, validation selection, cache export, and real 125-step versus 25-step timing. It writes `/kaggle/working/andi-e1-probe/report/probe.json`.

```python
import subprocess, sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROBE = True
branch = "codex/stack-05-e3"
with TemporaryDirectory() as work:
    repo = Path(work) / "ANDi"
    subprocess.run(["git", "clone", "--single-branch", "--branch", branch,
                    "https://github.com/ahmad-m-abbas/ANDi.git", str(repo)], check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    cmd = [sys.executable, "-u", "kaggle/one_cell.py", commit]
    subprocess.run(cmd + (["--probe"] if PROBE else []), cwd=repo, check=True)
```

After the preflight, set `PROBE = False` for the full experiment. The runner shares the inference shards across the available GPUs and runs the paired timing panel without concurrent inference. It also compares the 125-step baseline and selected 25-step schedule on two calibration subjects with a second noise seed. Results and figures are written to `/kaggle/working/andi-e1-full/report`. The new timestep caches are under `/kaggle/working/andi-e1-full` alongside the report. The full one-cell job must finish within Kaggle's session limit. The smaller jobs below can each be saved independently.

## Submit in order

Use `kaggle kernels status "$OWNER/$PREFIX-<name>"` to check a job. Advance to the next block once all its required inputs have completed successfully. Each job name is distinct so a successful shard remains available if another shard needs a retry.

Preflight and calibration

```bash
prepare smoke smoke
push_gpu smoke

# After smoke succeeds
prepare calibrate calibrate
push_gpu calibrate
```

Validation inference and selection

```bash
# After calibrate succeeds
for i in 0 1 2; do
  prepare validate "val-$i" --shard "$i" --shards 3 \
    --kernel-source "$OWNER/$PREFIX-calibrate"
  push_gpu "val-$i"
done

# After all three validation shards succeed
prepare select select --kernel-source "$OWNER/$PREFIX-calibrate" \
  --kernel-source "$OWNER/$PREFIX-val-0" \
  --kernel-source "$OWNER/$PREFIX-val-1" \
  --kernel-source "$OWNER/$PREFIX-val-2"
kaggle kernels push -p "$JOB_DIR/select"
```

Test inference and report

```bash
# After select succeeds and writes frozen.json
for i in {0..7}; do
  prepare test "test-$i" --shard "$i" --shards 8 \
    --kernel-source "$OWNER/$PREFIX-select"
  push_gpu "test-$i"
done

# After all eight test shards succeed
TEST_INPUTS=()
for i in {0..7}; do
  TEST_INPUTS+=(--kernel-source "$OWNER/$PREFIX-test-$i")
done
prepare score score --kernel-source "$OWNER/$PREFIX-select" "${TEST_INPUTS[@]}"
kaggle kernels push -p "$JOB_DIR/score"

# After score succeeds
kaggle kernels output "$OWNER/$PREFIX-score" -p "$JOB_DIR/results"
```

Each GPU shard saves the full-volume maps it computed. Validation and test shards also save any timesteps outside 75–199 in their `step-cache` output, with the packed masks and centered float16 foreground log residuals. Download those shard outputs or attach them as Kaggle inputs to reuse the caches. The scoring job saves `report.md`, `results.json`, the exact frozen configuration, calibration diagnostics and the accuracy-versus-speedup plot.

The shard counts above keep workloads and outputs separate under [Kaggle's per-session limits](https://www.kaggle.com/docs/notebooks). Submit fewer jobs at a time if your account queues GPU sessions. If a stage times out, increase that stage's shard count and use a fresh set of kernel names so the join receives a complete, non-overlapping partition.
