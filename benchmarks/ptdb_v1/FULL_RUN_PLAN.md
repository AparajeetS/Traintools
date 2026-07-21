# PTDB-1 full-run plan

Frozen on 2026-07-18 after the timing pilot and before development outcomes.

## Base matrix shards

The 81 instrumented base runs retain the datasets, architectures, seeds,
regimes, transforms, optimizers, and 25-epoch budget in `PROTOCOL.md`.

1. CIFAR-10: 27 instrumented runs in one T4 shard.
2. CIFAR-100: 27 instrumented runs in one T4 shard.
3. SVHN ResNet-18: nine instrumented runs in one T4 shard.
4. SVHN WRN-28-2: nine instrumented runs in one T4 shard.
5. SVHN ViT-Tiny/4: nine instrumented runs in one T4 shard.

The 18 development plain twins are the clean ResNet-18 and WRN-28-2 cells for
all three datasets and seeds. The timing pilot already includes two additional
ViT twins as implementation checks, but those do not replace the 18 declared
development pairs.

## Interpretation boundary

The base shards supply complete curves and example-level diagnostic records.
GNS batch-utility branches, plasticity task switches, injected-failure controls,
MBE placebo analysis, and the protected holdout remain separate registered
stages. Base-shard outcomes cannot alter their targets, comparators, or gates.

## Quota rule

Each shard must fit below Kaggle's 12-hour notebook limit according to pilot
telemetry. Infrastructure failures may be rerun only after a dated correction is
committed. Scientific failures, nulls, and sign reversals are never rerun away.

## Infrastructure amendment 1, 2026-07-22

The first CIFAR-10 shard reached Kaggle's 12-hour limit. Its log shows 29
executions completed and the thirtieth began, but Kaggle published no output
files for the cancelled script, so none of those rows can enter analysis. No
metric outcome was available when this amendment was made.

CIFAR-10 and CIFAR-100 are therefore split by architecture. CNN shards contain
nine instrumented runs and three clean plain twins; ViT shards contain nine
instrumented runs. Pilot and cancellation timing place each at roughly 4-5
hours. The three existing architecture-specific SVHN shards are unchanged.

Future kernels discover attached native dataset files by content under
`/kaggle/input` instead of assuming Kaggle's mount-directory spelling. Network
download remains a fallback. These changes affect only shard boundaries and
dataset staging; all scientific settings and gates remain frozen.
