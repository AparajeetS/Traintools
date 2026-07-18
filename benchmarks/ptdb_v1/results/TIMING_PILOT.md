# PTDB-1 timing pilot of record

Kaggle kernel: `aparajeetshadangi/phason-ptdb-1-timing-pilot`, version 5  
Execution date: 2026-07-18  
Hardware: one Tesla T4  
Software under test: `traintools==0.6.2` from PyPI

## Validation

- 12 of 12 executions completed with no error rows.
- Six of six plain/instrumented pairs had exact final parameter hashes.
- Maximum paired test-accuracy difference was 0.0.
- Every execution used the frozen three-epoch pilot budget.
- Manifest-declared artifact SHA-256 hashes match the downloaded files.

## Timing result

- Median instrumented/plain runtime ratio: 2.91.
- Mean instrumented time per epoch: 62.78 seconds.
- Projected core time for 81 instrumented 25-epoch development runs: 35.31
  single-T4 GPU-hours, before the separately declared plain twins and branch
  interventions.

The overhead is reported as measured, not hidden or normalized away. The full
matrix must be resumable and divided into sub-12-hour shards.

## Interpretation boundary

This pilot establishes package integration, non-interference for the six tested
pairs, artifact completeness, and a compute estimate. Its diagnostic values may
not validate a metric, alter a threshold, or modify a PTDB-1 success gate.
