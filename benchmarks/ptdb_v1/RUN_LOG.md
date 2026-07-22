# PTDB-1 run log

## CIFAR-10 combined development attempt

- Submitted: 2026-07-18 16:09:09 +05:30
- Kaggle kernel: `aparajeetshadangi/phason-ptdb-1-cifar-10-development`
- Kernel version: 1
- Source commit: `fec9ef564576c0904befce9d1dcaa7c1dd302e5a`
- Requested accelerator: `NvidiaTeslaT4`
- Protocol SHA-256: `5b01d1ca47b6ebca6601a9debc80b0edfb262c3ee6feeb8e0cc998b0ee92821e`
- Submission matrix: 27 instrumented runs and six plain twins, 25 epochs each
- Final Kaggle status: `CANCEL_ACKNOWLEDGED` after the 12-hour runtime ceiling

- Published output files: none.
- Log progress: 29 executions completed; execution 30 began but did not finish.
- Scientific rows admitted to analysis: zero.

Because Kaggle does not publish `/kaggle/working` for this cancelled script, the
partial CSV ledger is unavailable. The complete matrix must be rerun in smaller
architecture shards. This is an infrastructure rerun under the frozen protocol,
not a response to favorable or unfavorable metric results.

## CIFAR-10 ResNet-18 replacement shard

- Submitted: 2026-07-22 00:40:01 +05:30
- Kaggle kernel: `aparajeetshadangi/phason-ptdb-1-cifar-10-resnet18-development`
- Kernel version: 1
- Source commit: `8b7ce210b4eb5a288da5042c84d910cdb143371c`
- Requested accelerator: `NvidiaTeslaT4`
- Submission matrix: nine instrumented runs and three clean plain twins
- Epochs: 25 per execution
- Status after setup window: running

### Completion status, checked 2026-07-22 13:12 +05:30

- Kaggle status: `COMPLETE`.
- Frozen validator: passed every check.
- Completed executions: 12 of 12 with zero error rows.
- Epoch rows: 300 of 300.
- Exact plain/instrumented hashes: three of three.
- Maximum paired accuracy and loss differences: 0.0.
- Downloaded evidence: `results/cifar10_resnet18_v1/`.
- Frozen analysis: `results/analysis_cifar10_resnet18_v1/`.

## CIFAR-10 WRN-28-2 replacement shard

- Submitted: 2026-07-22 13:17:22 +05:30
- Kaggle kernel: `aparajeetshadangi/phason-ptdb-1-cifar-10-wrn28-2-development`
- Kernel version: 1
- Source commit: `8b7ce210b4eb5a288da5042c84d910cdb143371c`
- Requested accelerator: `NvidiaTeslaT4`
- Submission matrix: nine instrumented runs and three clean plain twins
- Epochs: 25 per execution

The Kaggle-created slug inserted a hyphen in `cifar-10`; the local metadata ID
was corrected after version 1 started. This pointer correction changed no code,
configuration, dataset, model, seed, regime, threshold, or success gate.

### Completion status, checked 2026-07-22

- Kaggle status: `COMPLETE`.
- Frozen validator: passed every check.
- Completed executions: 12 of 12 with zero error rows.
- Epoch rows: 300 of 300.
- Exact plain/instrumented hashes: three of three.
- Maximum paired accuracy and loss differences: 0.0.
- Downloaded evidence: `results/cifar10_wrn28_2_v1/`.
- Frozen analysis: `results/analysis_cifar10_wrn28_2_v1/`.
- Combined frozen analysis: `results/analysis_cifar10_resnet18_wrn28_2_v1/`.
