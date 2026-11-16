# Repository organization review — 16 November 2026

## Reference and scope

The mechanical layer starts at upstream/main `b30306f`, also origin/main at
review time. Published behavior is defined by the released source and
[Frotscher et al., ANDi](https://arxiv.org/pdf/2312.01904), especially Algorithm 1
and sections 4–5. The local `ANDi_Research_Experiments.pdf` (14 November 2026)
and baseline `results/reproduction/report.md` and `maps/manifest.json` were
reviewed. The latter artifacts are ignored local research outputs, not new
validation results. No AGENTS.md was found in the repository or its parent tree.

Preserve training pyramidal noise, Gaussian evaluation noise, reverse iteration
of indices 75–199 (stop 200 exclusive), geometric mean without added epsilon,
modality maximum, original normalization/filter/dilation and reporting.
Experiment-specific seed streams, stabilizers, score scales, masks, thresholds,
selection rules and output formats are retained separately. No dependency,
checkpoint format, W&B integration, new helper implementation or test suite is
added. Temporary review probes are not committed.

## Branch inventory and planned stack

All existing codex branches are ancestor snapshots of one history; none has a
unique divergent side commit. Remote discovery checked both origin and upstream.
Upstream has no codex branches. Local and origin main differ only by `8488314`
(research-plan documents and Ljubljana split).

| Old branch | Local tip | Remote tip before work | Unique work after preceding snapshot | Stack destination |
| --- | --- | --- | --- | --- |
| codex/brats21-mps-reproduction | 9123c83 | absent | e4cf6b2 MPS evaluator; 9123c83 E2 cached threshold scoring | 02 MPS reproduction; 03 E2 |
| codex/brats21-e1 | 5663769 | absent | e68a332 MPS E1; 5663769 diagnostic wording | 04 E1 |
| codex/brats21-e1-kaggle | eda35a1 | eda35a1 | b11372a CUDA stages; 0bb04a8 pinned model; eda35a1 one-cell runner | 04 E1 |
| codex/brats21-e3-kaggle | d3b9359 | d3b9359 | d3b9359 E3, E1 observer/scoring extensions, Kaggle support | 05 E3 |

1. `codex/stack-01-organization`: only shared organization against main.
2. `codex/stack-02-mps-reproduction`: existing e4cf6b2 evaluator port and config.
   This is a reproduction-support layer, not part of mechanical PR 1.
3. `codex/stack-03-e2-cached`: existing cached E2 analysis, not a new synthetic
   generator. The archived-cache workflow remains available with its original
   required inputs; completing end-to-end E2 generation is deferred.
4. `codex/stack-04-e1`: MPS and CUDA E1, including the later Kaggle improvements.
5. `codex/stack-05-e3`: E3 and only its E1/Kaggle extensions; includes the existing
   Ljubljana split from 8488314 because transfer depends on it.

E1 imports E2's threshold grid/scoring, so E2 precedes E1 despite experiment
numbering. E3 imports E1's inference and scoring. Shared ancestor changes land
once; earlier E1 snapshots are superseded by the combined E1 layer. Existing
branches/history remain intact. Research-plan PDF/LaTeX/image files from 8488314
stay on the old branches; they are not organization or experiment implementation.
E4/E5 exist only as proposals. Ignored data, weights, results and untracked Kaggle
outputs/.DS_Store are excluded.

## Duplicate review decisions

- Root modules forward to one implementation, rather than retaining copied
  bodies. Existing wildcard-export names remain available.
- `ddpm_mu_t` and `ddpm_mean_t` differ: the latter reconstructs/clips x0. Keep both.
- ANDi and DAE U-Nets differ in architecture, conditioning and checkpoint keys.
- `dice_stitch`, patient `dice`, threshold sweeps and experiment metrics differ in
  aggregation, input types, foreground masks and denominators. Keep them.
- `norm_tensor` normalizes the entire tensor; experiment normalization receives
  explicit extrema. Do not replace one with the other.
- Baseline gmean has no epsilon; E3 uses logaddexp with delta=1e-12 and scale
  floor=0.1. Keep baseline and calibrated aggregation separate.
- MPS inference replays historical reverse-order RNG draws; CUDA seeds each
  split/subject/timestep/batch. Preserve both implementations.
- E2 `split_ids`, E1 `subject_ids`, E3 `subjects` have different accepted inputs
  and CSV-column handling. Their similar bodies are not interchangeable.
- JSON writers differ in atomic writes, newline and NaN handling. Image save and
  upload differ in return/output behavior. Retain these implementations.

## Findings for separate logic work (not fixed)

1. Training resume reads `conf['ema']` but shipped `conf/train.yml` defines
   `ema_model`. Enabling train_continue with the shipped key fails with KeyError.
   Resolve the configuration contract separately, preserving old checkpoint files.
2. The MPS evaluator seeds once and keeps `size_splits: 155`; archived baseline
   manifests specify `seed + subject_index` and batches of 8. It cannot be claimed
   to regenerate those archives exactly. E1's archive replay deliberately uses
   batch 8 and per-subject seeds. Aligning the evaluator requires a logic/config
   decision outside this mechanical stack.
3. Cached E2 requires `results/cache/synthetic` and `results/cache/val`; no
   synthetic-cache generator exists in any reviewed branch. Do not claim a
   fresh-checkout end-to-end E2 reproduction. Recover/generate those artifacts
   in separate work; the cached scorer itself is retained unchanged.
4. E1 archive setup validates manifest batch=8, but setup leaves the default
   inference configuration's size_splits=155 untouched (E1 uses its own BATCH=8).
   Keep this distinction explicit when comparing command outputs.

## Verification boundaries

No repository test suite or CI checks are tracked. The organization layer was
checked by exact AST comparison of every moved function/class against upstream,
CPU preprocessing/postprocessing/metric and seeded-noise parity, legacy imports,
and strict loading of the real released EMA state dict plus identical CPU U-Net
forward output. Configuration paths, state-dict keys and save call sites are
unchanged. `environment.yml` is unchanged. Training runtime is blocked by missing
local wandb; no new dependencies are installed. Accelerator inference and full
cohort reruns are reported per PR rather than inferred from static checks.
