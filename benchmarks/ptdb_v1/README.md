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
