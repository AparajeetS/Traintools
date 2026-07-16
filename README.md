# traintools

<!-- mcp-name: io.github.aparajeets/traintools -->

[![CI](https://github.com/AparajeetS/Traintools/actions/workflows/ci.yml/badge.svg)](https://github.com/AparajeetS/Traintools/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/traintools.svg)](https://pypi.org/project/traintools/)
[![Python](https://img.shields.io/pypi/pyversions/traintools.svg)](https://pypi.org/project/traintools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Paper-backed ML training diagnostics for PyTorch. Small tools that answer
practical questions while a run is still alive.

[Source on GitHub](https://github.com/AparajeetS/Traintools) | [PyPI](https://pypi.org/project/traintools/)

Have a real run or false alarm to share? Use the
[diagnostic report template](https://github.com/AparajeetS/Traintools/issues/new?template=diagnostic-report.yml).

```bash
pip install traintools
```

Not sure which tool fits the problem?

```bash
traintools recommend "my loss became NaN and gradients explode"
traintools integration gradient-health --framework pytorch
```

Agents can use the JSON CLI, [the agent guide](https://github.com/AparajeetS/Traintools/blob/main/AGENTS.md), [llms.txt](https://github.com/AparajeetS/Traintools/blob/main/llms.txt),
or the optional local MCP server:

```bash
pip install "traintools[mcp]"
traintools-mcp
```

The same local, read-only server is discoverable through the official MCP
Registry and can be launched without a persistent install:

```bash
uvx --with mcp traintools mcp
```

## Tools

| Tool | Question it answers |
|---|---|
| **Gradient Noise Scale (GNS)** | Is my batch size wasting compute? |
| **GradientAccumulationGNS** | Can I get GNS for free during gradient accumulation? |
| **PlasticityProbe** | Is my network losing the ability to learn? |
| **TrainGuard** | Should I stop training yet? |
| **BatchInspector** | Is this batch broken, imbalanced, or out of scale? |
| **GradientHealthMonitor** | Are gradients finite, clipped, vanished, exploded, or too large for the weights? |
| **ExampleDynamicsTracker** | Which examples are forgotten, hard, ambiguous, or likely mislabeled? |
| **GradientConfusionMonitor** | Are micro-batch gradients fighting each other and slowing SGD? |
| **AUMTracker** | Which examples look mislabeled by margin dynamics? |
| **EL2NTracker** | Which examples are important or pruneable early in training? |
| **NeuralCollapseMonitor** | Has the classifier entered neural-collapse geometry? |

## Quick Start

```python
from traintools import BatchInspector, GradientHealthMonitor
from traintools.callbacks.pytorch import TraintoolsTracker

tracker = TraintoolsTracker(model, loss_fn)
batch_inspector = BatchInspector(expected_num_classes=10)
grad_health = GradientHealthMonitor(max_grad_norm=1.0)

for step, (x, y) in enumerate(dataloader):
    batch_report = batch_inspector.inspect(x, y, step=step)
    if not batch_report.ok:
        print(batch_report)

    loss = loss_fn(model(x), y)
    loss.backward()

    grad_report = grad_health.inspect(model, step=step, lr=optimizer.param_groups[0]["lr"])
    if not grad_report.ok:
        print(grad_report)

    optimizer.step()
    optimizer.zero_grad()

    decision = tracker.step(step=step, inputs=x, targets=y, val_loss=val_loss)
    if decision and decision.should_stop:
        break
```

HuggingFace Trainer:

```python
from transformers import Trainer
from traintools.callbacks.huggingface import TraintoolsCallback

trainer = Trainer(model=model, ..., callbacks=[TraintoolsCallback()])
```

## Gradient Noise Scale

GNS is the ratio of per-example gradient variance to gradient signal:

```text
GNS = tr(Sigma) / ||G||^2
```

It estimates the critical batch size B*: the point where larger batches stop
buying much more optimization progress.

- `GNS > B`: under-batched, larger batches can help
- `GNS < B`: over-batched, the batch may be larger than needed
- `GNS ~= B`: near the efficient frontier

`traintools` uses the unbiased estimators from McCandlish et al. 2018
(Bessel-corrected variance, bias-corrected signal) and tracks GNS as
`EMA(tr(Sigma)) / EMA(||G||^2)`.

```text
[step 500] GNS=5010.7 (EMA)  critical_batch=5011  current=64  regime=under-batched
  > Batch size 64 is ~78x below the critical batch (~5011). Larger batches would give cleaner gradients per step.
```

### Free GNS During Gradient Accumulation

If you already use gradient accumulation, the per-micro-batch gradients you
compute anyway are exactly the samples GNS needs.

```python
from traintools import GradientAccumulationGNS

gns = GradientAccumulationGNS(model, micro_batch_size=B_micro)

for step in range(num_steps):
    for xm, ym in micro_batches:
        (loss_fn(model(xm), ym) / accum_steps).backward()
        gns.record_microbatch()
    optimizer.step()
    result = gns.compute(step=step)
    optimizer.zero_grad()
    gns.reset_accumulation()
```

## PlasticityProbe

PlasticityProbe measures activations directly:

- dormant unit fraction: units whose activation is near zero for every input
- feature effective rank: normalized effective rank of the activation covariance

Those are combined into a plasticity score in `[0, 1]`.

```text
[step 200] Plasticity Score: 0.706
  All layers healthy.
```

## TrainGuard

TrainGuard fits a power-law or exponential curve to validation loss, bootstraps
uncertainty, and only stops when continuing looks unlikely to matter.

```text
[step 400] STOP
  current loss: 0.6536
  predicted final: 0.6119
  expected improvement: 0.0417 (90% CI: [0.0012, 0.0821])
  estimated plateau at step: 3200
  reason: No improvement in 300 steps (best=0.6350 at step 93).
```

## BatchInspector

BatchInspector catches bad tensors and labels before they quietly poison a run.

```python
from traintools import BatchInspector

inspector = BatchInspector(expected_num_classes=10, max_abs_value=1e4)
report = inspector.inspect(inputs=x, targets=y, step=step)
if not report.ok:
    print(report)
```

It checks for empty tensors, NaNs/infs, extreme scales, constant tensors, labels
outside the expected class range, and severe batch imbalance.

## GradientHealthMonitor

GradientHealthMonitor is called after `backward()` and before `optimizer.step()`.

```python
from traintools import GradientHealthMonitor

monitor = GradientHealthMonitor(max_grad_norm=1.0)
loss.backward()
report = monitor.inspect(model, step=step, lr=optimizer.param_groups[0]["lr"])
if not report.ok:
    print(report)
```

It reports global and per-layer gradient norms, non-finite gradients, likely
vanishing/exploding gradients, clipping coefficient, and update-to-weight ratio.

## ExampleDynamicsTracker

ExampleDynamicsTracker implements two underused training-dynamics probes:

- example forgetting events from Toneva et al. 2019
- dataset-cartography-style confidence and variability from Swayamdipta et al. 2020

Use stable dataset ids, logits, and labels during a normal classification run.

```python
from traintools import ExampleDynamicsTracker

dynamics = ExampleDynamicsTracker()

for step, (ids, x, y) in enumerate(dataloader):
    logits = model(x)
    loss = loss_fn(logits, y)
    dynamics.update(ids, logits, y, step=step)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(dynamics.summary())
print("likely noisy or brittle:", [ex.example_id for ex in dynamics.most_forgotten(20)])
print("ambiguous:", [ex.example_id for ex in dynamics.cartography_region("ambiguous")])
```

A forgetting event is a transition from correct classification to incorrect
classification for the same example. Repeatedly forgotten examples are often
ambiguous, mislabeled, or distribution-edge cases. Unforgettable examples can be
useful candidates for pruning or curriculum experiments.

## GradientConfusionMonitor

GradientConfusionMonitor estimates whether micro-batch gradients are aligned or
fighting each other, following the gradient-confusion idea from Sankararaman et
al. 2019.

```python
from traintools import GradientConfusionMonitor

confusion = GradientConfusionMonitor(n_splits=4)
report = confusion.estimate(model, loss_fn, x, y, step=step)
if not report.ok:
    print(report)
```

It reports mean/min/max pairwise gradient cosine, the fraction of negative
gradient pairs, and a compact conflict score. High conflict can point to noisy
labels, incompatible samples, depth/initialization issues, or a need for a
different batching/curriculum strategy.

## AUMTracker

AUMTracker implements the Area Under the Margin statistic from Pleiss et al.
2020. For each example, it averages:

```text
true_class_logit - max(other_class_logits)
```

Low-AUM examples are candidates for label audit or ambiguity review.

```python
from traintools import AUMTracker

aum = AUMTracker(low_aum_threshold=0.0)
for step, (ids, x, y) in enumerate(dataloader):
    logits = model(x)
    aum.update(ids, logits, y, step=step)

print([ex.example_id for ex in aum.lowest_aum(20)])
```

## EL2NTracker

EL2NTracker implements the cheap example-importance score from Paul et al.
2021:

```text
||softmax(logits) - one_hot(label)||_2
```

High EL2N examples tend to be important, hard, noisy, or distribution-edge
examples. Low EL2N examples can be candidates for data-pruning experiments.

```python
from traintools import EL2NTracker

el2n = EL2NTracker()
el2n.update(ids, logits, y, step=step)
important = el2n.highest(100)
prune_candidates = el2n.lowest(100)
```

## NeuralCollapseMonitor

NeuralCollapseMonitor measures late-stage classifier geometry from Papyan, Han,
and Donoho 2020:

- NC1: within-class feature variability relative to between-class variability
- NC2: class means approaching simplex ETF geometry
- NC3: classifier weights aligning with class means, when weights are provided
- NCC accuracy: nearest-class-center accuracy

```python
from traintools import NeuralCollapseMonitor

collapse = NeuralCollapseMonitor()
report = collapse.measure(features, labels, classifier_weight=model.fc.weight)
print(report)
```

## Installation

```bash
# Core: PyTorch only
pip install traintools

# Curve fitting and plotting helpers
pip install traintools[full]

# HuggingFace Trainer integration
pip install traintools[hf]

# Local MCP server for compatible AI clients
pip install traintools[mcp]

# Development
pip install -e ".[dev]"
```

## Project Status

`traintools` is alpha software. The diagnostics are intentionally small and
well-tested, but thresholds are heuristics and should be interpreted as training
signals, not automatic truth. Bug reports, benchmark traces, and real-world
failure cases are especially welcome.

Problem-oriented guides live in [the documentation](https://github.com/AparajeetS/Traintools/tree/main/docs/problems). Diagnostic
objects can be written as versioned JSON with `write_json_report`.

## References

- McCandlish, Kaplan, Amodei et al. 2018. *An Empirical Model of Large-Batch Training*.
- Dohare, Sutton et al. 2024. *Loss of plasticity in deep continual learning*.
- Lyle et al. 2023. *Understanding Plasticity in Neural Networks*.
- Domhan et al. 2015. *Speeding up automatic hyperparameter optimization of DNNs by extrapolation of learning curves*.
- Toneva et al. 2019. *An Empirical Study of Example Forgetting during Deep Neural Network Learning*.
- Swayamdipta et al. 2020. *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics*.
- Sankararaman et al. 2019. *The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent*.
- Pleiss et al. 2020. *Identifying Mislabeled Data using the Area Under the Margin Ranking*.
- Paul, Ganguli, and Dziugaite 2021. *Deep Learning on a Data Diet: Finding Important Examples Early in Training*.
- Papyan, Han, and Donoho 2020. *Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training*.

## License

MIT
