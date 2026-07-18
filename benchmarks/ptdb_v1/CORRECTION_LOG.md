# PTDB-1 correction log

## Timing pilot version 1, 2026-07-18

Kaggle terminated before any training cell began. Script kernels upload the
declared `code_file` but did not place the adjacent `benchmark_config.json` at
`/kaggle/src/benchmark_config.json`. The runner raised `FileNotFoundError` while
loading configuration.

Correction: embed the byte-equivalent frozen configuration and the frozen
protocol SHA-256 directly in the script. The external JSON remains as a human-
readable checked copy, and local runs fail if it differs from the embedded
configuration.

No dataset, model, seed, intervention, epoch budget, comparator, threshold, or
success gate changed. Version 1 produced no outcome data and consumed only
environment-setup time.

The Kaggle-created slug normalized `PTDB-1` to `phason-ptdb-1-timing-pilot`.
The local metadata ID was updated to that existing slug after the first update
request returned HTTP 409; this changes no executable or research setting.
