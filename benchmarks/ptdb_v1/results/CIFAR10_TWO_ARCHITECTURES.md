# PTDB-1 CIFAR-10 two-architecture development result

This comparison combines the validated ResNet-18 and WRN-28-2 shards under the
frozen PTDB-1 analysis. It contains 24 complete executions, 600 epoch rows, six
exact clean plain/instrumented pairs, and 810,000 example-level score rows.

## Aggregate results

| Score | ResNet-18 AUROC | WRN-28-2 AUROC | Combined AUROC |
|---|---:|---:|---:|
| EL2N | 0.9932 | 0.9913 | 0.9922 |
| Mean confidence | 0.9895 | 0.9855 | 0.9875 |
| AUM | 0.9885 | 0.9859 | 0.9872 |
| Mean loss | 0.9792 | 0.9750 | 0.9771 |
| Deterministic random | 0.4986 | 0.5022 | 0.5004 |
| Forgetting count | 0.3361 | 0.3270 | 0.3316 |

The predeclared AUM-minus-mean-loss contrast is `0.01011`, with four-cluster
bootstrap interval `[0.00080, 0.01942]`. AUM beats mean loss in all twelve noisy
runs across the two architectures.

## Replicated ordering

The architecture replication preserves a sharper qualitative result:

- AUM exceeds EL2N in all six symmetric-noise runs.
- EL2N exceeds AUM in all six class-conditional-noise runs.
- Forgetting count is below random in all twelve runs under the declared direction.
- All six clean instrumented twins exactly match their plain counterparts.

The result argues against a universal metric leaderboard and supports the
task-conditioned metric-selection framing: which diagnostic is strongest
depends here on the corruption mechanism, and that dependence reproduces across
two CNN architectures.

## Boundary

This remains one dataset, two related convolutional architectures, synthetic
20% label corruption, and four dataset-model-regime clusters. It does not
establish natural-label-noise performance, cross-dataset generalization, MBE's
broader selector claim, language-model applicability, or protected-holdout
success. Gradient-norm and stopping-policy claims remain withheld.
