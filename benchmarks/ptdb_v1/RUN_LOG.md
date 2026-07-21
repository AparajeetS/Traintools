# PTDB-1 run log

## CIFAR-10 development shard

- Submitted: 2026-07-18 16:09:09 +05:30
- Kaggle kernel: `aparajeetshadangi/phason-ptdb-1-cifar-10-development`
- Kernel version: 1
- Source commit: `fec9ef564576c0904befce9d1dcaa7c1dd302e5a`
- Requested accelerator: `NvidiaTeslaT4`
- Protocol SHA-256: `5b01d1ca47b6ebca6601a9debc80b0edfb262c3ee6feeb8e0cc998b0ee92821e`
- Submission matrix: 27 instrumented runs and six plain twins, 25 epochs each
- Status after setup window: running

The Kaggle-created slug inserted a hyphen in `cifar-10`; the local metadata ID
was corrected after version 1 started. This pointer correction changes no code,
configuration, dataset, model, seed, regime, threshold, or success gate.

### Completion status, checked 2026-07-22

- Kaggle status: `CANCEL_ACKNOWLEDGED` after the 12-hour runtime ceiling.
- Published output files: none.
- Log progress: 29 executions completed; execution 30 began but did not finish.
- Scientific rows admitted to analysis: zero.

Because Kaggle does not publish `/kaggle/working` for this cancelled script, the
partial CSV ledger is unavailable. The complete matrix must be rerun in smaller
architecture shards. This is an infrastructure rerun under the frozen protocol,
not a response to favorable or unfavorable metric results.
