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

## Timing pilot version 2, 2026-07-18

Kaggle allocated a Tesla P100 (compute capability 6.0). The dependency resolver
for `pip install traintools==0.6.2` replaced Kaggle's preinstalled PyTorch with a
build supporting compute capability 7.0 and newer. All 12 executions therefore
failed while moving the model to CUDA with `no kernel image is available for
execution on the device`; no epoch began and no outcome data were produced.

Correction: install the same public TrainTools release with `--no-deps`, thereby
preserving Kaggle's platform-compatible PyTorch build, and run a real CUDA tensor
operation before dataset setup so an incompatible environment fails immediately.

Version 2 also showed that downloading the official archives during the run was
too slow for a useful timing pilot and caused dataset caches to be captured as
notebook output. The exact native dataset formats are now attached from public
Kaggle mirrors and staged under `/kaggle/temp`; torchvision's built-in integrity
checks still validate the expected files. Network download remains a fallback
outside Kaggle. Temporary data are excluded from notebook output.

No dataset identity, split, transform, model, seed, intervention, epoch budget,
comparator, threshold, or success gate changed. These infrastructure corrections
were recorded before version 3 was submitted.

## Timing pilot version 3, 2026-07-18

The new CUDA preflight failed in about 17 seconds, before dataset staging or any
training. It isolated a second environment problem: Kaggle's then-current base
image itself contained PyTorch 2.10.0+cu128, whose binary supported capabilities
7.0 through 12.0, while the assigned Tesla P100 has capability 6.0. Thus the
version 2 failure cannot be attributed solely to TrainTools dependency resolution;
the platform's default GPU and framework image were independently incompatible.

Correction: submit version 4 with Kaggle CLI's explicit
`--accelerator NvidiaTeslaT4` option. The T4 has compute capability 7.5 and is
within the capability range reported by the unchanged framework build. The CUDA
preflight remains mandatory and the public package remains installed with
`--no-deps`.

No outcome data were produced, and no benchmark setting or success gate changed.

## Timing pilot version 4, 2026-07-18

The explicit T4 request was honored and the CUDA preflight passed. Training did
not begin because this Kaggle script environment had no `/kaggle/temp` directory;
the cache-path fallback consequently resolved to the read-only `/kaggle/src/data`.
All 12 executions stopped while creating that directory.

Correction: use `/tmp/ptdb_data`, the container's writable ephemeral filesystem,
whenever `/kaggle` is present. This cache is not a notebook output and is discarded
with the worker. Local execution continues to use the repository-ignored `data/`
directory.

No outcome data were produced, and no benchmark setting or success gate changed.

## Timing pilot version 5, 2026-07-18

Version 5 completed on one Tesla T4. The frozen validator accepted all 12
executions and all six plain/instrumented pairs with no error rows. All six pairs
had exact final parameter hashes and zero test-accuracy difference. This version
is the timing pilot of record; its metric values remain non-confirmatory.
