# PTDB-1 CIFAR-10 / ResNet-18 development result

Kaggle kernel: `aparajeetshadangi/phason-ptdb-1-cifar-10-resnet18-development`, version 1  
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

## Label-noise detection

Results aggregate three seeds and the two frozen 20% corruption regimes. The
intervals are the precommitted 10,000-draw cluster bootstrap over regime cells;
with one architecture and only two clusters, they must not be read as broad
cross-architecture uncertainty.

| Score | AUROC | Average precision | Precision at 20% |
|---|---:|---:|---:|
| AUM | 0.9885 | 0.9285 | 0.9143 |
| EL2N | 0.9932 | 0.9731 | 0.9274 |
| Mean confidence | 0.9895 | 0.9386 | 0.9166 |
| Mean loss | 0.9792 | 0.8727 | 0.8749 |
| Forgetting count | 0.3361 | 0.1728 | 0.0896 |
| Deterministic random | 0.4986 | 0.1996 | 0.1974 |

AUM minus mean-loss AUROC was `0.00935`, with frozen cluster-bootstrap interval
`[0.00083, 0.01786]`. AUM exceeded the protocol's `0.85` dataset gate on this
slice and precision at the noise rate was more than four times prevalence.

Seed-level AUM AUROC ranged from `0.9935` to `0.9942` under symmetric noise and
from `0.9827` to `0.9834` under class-conditional noise.

## What this suggests

AUM is a strong label-noise detector in this setting and adds a small advantage
over accumulated mean loss. It is not the winner among all tested scores: EL2N
is slightly stronger here, and mean confidence is effectively tied with AUM.
Forgetting count points in the wrong direction for this target. That combination
supports PTDB-1's task-conditioned metric-selection framing more than a claim
that one training metric is universally best.

## Cost

Median instrumented/plain runtime ratio was `3.01`; p90 was `3.34`. Mean
instrumented time per epoch was `62.06` seconds. This overhead is material and
will remain part of the public result rather than being normalized away.

No threshold, comparator, intervention, analysis rule, or holdout choice was
changed after observing this shard.
