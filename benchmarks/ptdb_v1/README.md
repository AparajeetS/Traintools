# PTDB-1

The Phason Training Diagnostics Benchmark tests TrainTools against declared
operational targets on real image-classification workloads.

`PROTOCOL.md` is frozen before the timing pilot. The pilot is deliberately
non-confirmatory: it validates runtime, GPU memory, deterministic pairing,
package integration, and output completeness. It cannot tune metric thresholds.

## Pilot

The Kaggle kernel source is in `kaggle/pilot/`.

Local smoke test:

```powershell
$env:PTDB_USE_LOCAL='1'
$env:PTDB_SMOKE='1'
python benchmarks/ptdb_v1/kaggle/pilot/ptdb_v1.py
```

Validate downloaded Kaggle output:

```powershell
python benchmarks/ptdb_v1/kaggle/pilot/validate_pilot.py PATH_TO_OUTPUT
```

The full benchmark is launched only after the pilot establishes a quota-safe
shard plan. Every full shard uses the same run and artifact schemas.

## Pilot Of Record

Kaggle version 5 completed on a Tesla T4 on 2026-07-18. The downloaded immutable
artifacts and checksum validation are in `results/timing_pilot_v5/`; the concise
interpretation is in `results/TIMING_PILOT.md`.

The first validated 25-epoch development result, CIFAR-10 with ResNet-18, is
summarized in `results/CIFAR10_RESNET18.md`. It is development evidence, not the
protected holdout.
