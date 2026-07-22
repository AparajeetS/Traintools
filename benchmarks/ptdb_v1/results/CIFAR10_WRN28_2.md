# PTDB-1 CIFAR-10 / WRN-28-2 development result

Kaggle kernel: `aparajeetshadangi/phason-ptdb-1-cifar-10-wrn28-2-development`, version 1  
Execution date: 2026-07-22  
Hardware: one Tesla T4  
Software under test: `traintools==0.6.2` from PyPI  
Status: validated development evidence; protected holdout remains sealed

## Integrity

- 12 of 12 executions and 300 of 300 epoch rows completed.
- No error rows were recorded.
- All three clean plain/instrumented pairs had exact final parameter hashes.
- Maximum paired test-accuracy and loss differences were both 0.0.
- Every manifest-declared artifact SHA-256 matched the downloaded file.
- An independent standard-library audit recomputed all rankings from 405,000
  example rows; maximum discrepancy was `5.45e-15`.

## Label-noise detection

Results aggregate three seeds and the two frozen 20% corruption regimes. The
intervals use the precommitted 10,000-draw cluster bootstrap. With one
architecture and only two regime clusters, they do not represent broad
cross-architecture uncertainty.

| Score | AUROC | Average precision | Precision at 20% |
|---|---:|---:|---:|
| EL2N | 0.9913 | 0.9644 | 0.9176 |
| AUM | 0.9859 | 0.9111 | 0.9023 |
| Mean confidence | 0.9855 | 0.9111 | 0.9003 |
| Mean loss | 0.9750 | 0.8522 | 0.8573 |
| Deterministic random | 0.5022 | 0.2016 | 0.2031 |
| Forgetting count | 0.3270 | 0.1732 | 0.0797 |

AUM minus mean-loss AUROC was `0.01087`, with frozen cluster-bootstrap interval
`[0.00076, 0.02098]`. AUM exceeded mean loss in all six noisy runs.

AUM exceeded EL2N in all three symmetric-noise runs. EL2N exceeded AUM in all
three class-conditional runs and therefore had the higher aggregate AUROC.
Forgetting count was below random in all six runs under the prospectively
declared score direction.

## Operational findings

Median instrumented/plain runtime ratio was `3.94`; p90 was `4.04`. This is the
deliberately probe-heavy PTDB configuration, not a minimal integration estimate.

Fifteen epoch rows contained infinite mean gradient norms while primary losses,
accuracies, and final models remained finite. TrainGuard emitted three stop
flags outside the frozen decision epochs. Two flagged runs subsequently improved
validation loss by `0.0691` and `0.0432`; stopping utility is therefore withheld
pending the registered policy comparison.

No threshold, comparator, intervention, analysis rule, or holdout choice was
changed after observing this shard.
