# Phason Training Diagnostics Benchmark v1 (PTDB-1)

Status: prospectively frozen before the timing pilot, 2026-07-18.

## Purpose

PTDB-1 tests whether the public TrainTools package supplies useful operational
information on real image-classification workloads. It does not estimate an
"average ML run" and does not treat one diagnostic as universally good or bad.
Each diagnostic is evaluated only against the decision target it was designed
to address.

The timing pilot may change epoch budgets or shard boundaries only to fit the
available GPU quota. It may not change datasets, model families, interventions,
comparators, estimands, success gates, or the protected holdout in response to
metric results.

## Public Software Boundary

- `traintools==0.6.2` from PyPI is the tested package.
- The benchmark runner and analysis are versioned separately from the package.
- Torch, torchvision, CUDA, GPU, Python, and package versions are recorded.
- Every output row contains a run ID, seed, dataset, architecture, intervention,
  git/script hash, timing, and completion status.

## Development Matrix

Full-data training is used unless a run is explicitly marked `timing_pilot`.

| Dimension | Frozen levels |
|---|---|
| Datasets | CIFAR-10, CIFAR-100, SVHN |
| Architectures | ResNet-18, WRN-28-2, ViT-Tiny/4 |
| Seeds | 11, 23, 37 |
| Data regimes | clean, 20% symmetric label noise, 20% class-conditional label noise |
| Optimizer | SGD with momentum 0.9 for CNNs; AdamW for ViT |
| Schedule | cosine decay, no outcome-dependent early termination |
| Primary budget | 25 epochs; timing pilot uses 3 epochs |

This gives 81 instrumented development runs. Eighteen prospectively selected
clean cells receive plain-PyTorch twins for non-interference analysis.

## Timing Pilot

The six fixed pilot cells are:

1. CIFAR-10 / ResNet-18 / seed 11 / clean;
2. CIFAR-10 / ViT-Tiny / seed 11 / symmetric noise;
3. CIFAR-100 / WRN-28-2 / seed 23 / clean;
4. CIFAR-100 / ResNet-18 / seed 23 / class-conditional noise;
5. SVHN / WRN-28-2 / seed 37 / clean;
6. SVHN / ViT-Tiny / seed 37 / symmetric noise.

Pilot outputs can establish implementation validity and compute projections.
They cannot establish diagnostic accuracy or tune thresholds.

## Diagnostic Estimands And Comparators

### AUM and example dynamics

Target: whether the observed training label was deliberately corrupted.

Comparators: EL2N, mean per-example loss, forgetting count, mean true-class
confidence, and deterministic random ranking.

Primary outcomes: AUROC, average precision, and precision at the known noise
rate. Results are first computed within run, then aggregated by dataset-model
cell. Individual examples do not count as independent task families.

Gate: AUM AUROC at least 0.85 in every development dataset, precision at the
noise rate at least twice prevalence, and pooled cluster-bootstrap performance
above per-example loss.

### Gradient Noise Scale

Target: empirical benefit of increasing batch size at a common checkpoint.

At frozen checkpoints, training branches use batches 32, 128, 512, and 1024
for three continuation epochs. Outcomes are loss reduction per optimizer step,
loss reduction per example, throughput, and time to a fixed loss.

Comparators: batch 128 for every task, random eligible batch, and the empirical
oracle. Gate: task-cell Spearman correlation at least 0.60 between predicted
benefit and empirical benefit, with lower selection regret than fixed 128 and
random selection.

### Plasticity

Target: relearning speed after a task switch. Models receive either ordinary
pretraining or corrupted-label preconditioning before the same clean switch.

Comparators: current training loss, validation loss, gradient norm, activation
sparsity, dormant fraction alone, and effective rank alone.

Gate: positive leave-one-cell-out incremental utility beyond loss and accuracy,
with a task-cluster bootstrap confidence interval excluding zero and stable
direction in CNN and transformer groups.

### TrainGuard

Target: future validation-loss improvement from decision points at 25%, 40%,
and 50% of the fixed training budget. Complete curves are always run; stopping
is evaluated retrospectively.

Comparators: fixed budget, ordinary patience, recent validation-loss slope,
exponential extrapolation, and oracle stopping.

Gate: at least 20% median compute saved with at most 1% median final-loss regret,
and no catastrophic protected-holdout stop.

### Failure detection

Injected families: NaN, infinity, exploding gradients, vanishing gradients,
invalid labels, extreme imbalance, constant inputs, and extreme input scale.
Clean controls are matched by dataset, model, and seed.

Gate: sensitivity at least 95% and false-positive rate at most 5%, with Wilson
intervals and all individual warnings retained.

## Non-Interference

Plain and instrumented twins use identical initial state, batches, augmentation
seeds, optimizer state, AMP state, and deterministic settings.

Gate: exact final parameter hashes in at least 95% of pairs, maximum absolute
accuracy difference at most 0.2 percentage points, and no systematic paired
validation-loss shift. Median, p90, and per-diagnostic wall-clock and peak-VRAM
overhead are reported even if they are unfavorable.

## MBE Audit Layer

Metrics are never pooled across incompatible targets. MBE controls the ordinary
baseline information available for each declared decision. Inference clusters
by dataset-model-intervention cell.

A panel of 100 deterministic placebo metrics replaces reliance on one random
column. The observed placebo rejection rate and confidence interval are reported
against the nominal 5% level. A single favorable placebo realization cannot
validate the audit.

## Protected Holdout

STL-10 with ConvNeXt-Tiny is not executed until development code, thresholds,
tables, and analysis hashes are frozen. It uses seeds 11, 23, and 37 under clean
and 20% symmetric-noise regimes. No holdout result can modify a primary rule.

## Analysis Rules

- Primary uncertainty uses cluster bootstrap over run cells, not examples.
- Seed-level and cell-level results remain visible.
- Errors and incomplete rows are never silently dropped.
- No rerun is permitted for an unfavorable result. Reruns are allowed only for
  documented infrastructure or implementation failures.
- All deviations receive a dated amendment before replacement outcomes run.

## Publication Rule

Only gates satisfied on development and the untouched holdout may become a
headline website claim. Failed gates, null results, sign reversals, resource
costs, and the complete run ledger remain public.

