# PTDB-1 base-matrix analysis plan

Frozen before any 25-epoch development shard completed.

## Units and uncertainty

- A run is one dataset-model-regime-seed combination.
- The cluster unit is dataset-model-regime; seeds are repeated measurements.
- Cluster bootstrap intervals use 10,000 draws and seed 20260718.
- Cell means receive equal weight so larger datasets do not dominate inference.
- Missing or failed runs invalidate completeness; they are not silently removed.

## Label-noise table

The six prospectively directed scores are AUM (lower is suspicious), EL2N,
forgetting count, mean confidence (lower is suspicious), mean loss, and the
deterministic random score. Per-run AUROC, average precision, and precision at
the known 20% noise rate are retained. Dataset tables report equal-cell means
and cluster-bootstrap intervals. The primary contrast is AUM AUROC minus mean-
loss AUROC.

## Non-interference and overhead

All declared plain/instrumented pairs remain visible. Report exact-hash rate,
maximum absolute accuracy and loss differences, and median and p90 runtime
ratios. The protocol gates are evaluated without tolerance changes.

## TrainGuard targets from complete curves

Decision epochs are 6, 10, and 13 of 25. Complete curves are always used. For
each decision, realized future improvement is current validation loss minus the
minimum later validation loss. The emitted table includes TrainGuard's decision,
prediction and interval plus a three-point validation-loss slope comparator.
Stopping-policy analysis is a separate frozen stage because ordinary-patience,
extrapolation, oracle, compute-saved, and regret calculations must be compared
jointly.

## Boundaries

Base-matrix plasticity and GNS values are descriptive probes only. Their primary
utility claims require the registered task-switch and batch-branch experiments.
No base result opens the STL-10/ConvNeXt protected holdout.
