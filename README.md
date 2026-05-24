# traintools

Training diagnostics for PyTorch. Three tools, two lines of integration.

```
pip install traintools[full]
```

## What it does

| Tool | Question it answers |
|---|---|
| **GNS** — Gradient Noise Scale | Is my batch size wasting compute? |
| **PlasticityProbe** | Is my network losing the ability to learn? |
| **TrainGuard** | Should I stop training yet? |

## Quick start

```python
from traintools.callbacks.pytorch import TraintoolsTracker

tracker = TraintoolsTracker(model, loss_fn)

for step, (x, y) in enumerate(dataloader):
    loss = loss_fn(model(x), y)
    loss.backward()
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

## Gradient Noise Scale (GNS)

GNS is the ratio of per-example gradient variance to gradient signal:

```
GNS = tr(Σ) / ||G||^2
```

It equals the critical batch size B* — the point of diminishing returns from
larger batches.

- `GNS > B`  → **under-batched**: gradient too noisy, larger batches help
- `GNS < B`  → **over-batched**: batch larger than needed, shrink it and save compute
- `GNS ≈ B`  → **optimal**

`traintools` uses the *unbiased* estimators from McCandlish et al. 2018
(Bessel-corrected variance, bias-corrected signal) and tracks GNS as the ratio
of two separate exponential moving averages — the stable estimator the paper
recommends. (Naive single-shot estimates are biased low by ~2x and far too
noisy to act on.)

```
[step 500] GNS=5010.7 (EMA)  critical_batch=5011  current=64  regime=under-batched
  > Batch size 64 is ~78x below the critical batch (~5011). Larger batches would give cleaner gradients per step.
```

### Free GNS during gradient accumulation

If you already use gradient accumulation, GNS costs **zero extra forward/backward
passes** — the per-micro-batch gradients you compute anyway are exactly the
samples GNS needs. Every other GNS implementation pays for extra passes.

```python
from traintools import GradientAccumulationGNS

gns = GradientAccumulationGNS(model, micro_batch_size=B_micro)

for step in range(num_steps):
    for micro in micro_batches:
        (loss_fn(model(xm), ym) / accum_steps).backward()
        gns.record_microbatch()          # after each micro-batch backward
    optimizer.step()
    result = gns.compute(step=step)      # GNSResult, free
    optimizer.zero_grad()
    gns.reset_accumulation()
```

Reference: McCandlish et al. 2018, *An Empirical Model of Large-Batch Training*.

## PlasticityProbe

Networks lose plasticity over long training runs or repeated fine-tuning. The
failure shows up in the *representations*, so PlasticityProbe measures the
activations directly (not weight matrices), matching the operational definitions
in the loss-of-plasticity literature:

- **Dormant unit fraction** — units whose activation is ~0 for every input,
  attributed to the activation module that produced them
- **Feature effective rank** — effective rank of the activation covariance,
  normalised to [0,1]; low rank = representational collapse

Combined into a **Plasticity Score ∈ [0, 1]** (1 = fully plastic, 0 = dead).

```
[step 200] Plasticity Score: 0.706
  All layers healthy.
```

References: Dohare & Sutton et al. 2024 (*Nature* 632:768), *Loss of plasticity
in deep continual learning*; Lyle et al. 2023, *Understanding Plasticity in
Neural Networks*.

## TrainGuard

Fits a power-law or exponential curve to your validation loss history, bootstraps
uncertainty over the fit, and predicts whether continuing training is worth it:

```
[step 400] STOP
  current loss: 0.6536
  predicted final: 0.6119
  expected improvement: 0.0417 (90% CI: [0.0012, 0.0821])
  estimated plateau at step: 3200
  reason: No improvement in 300 steps (best=0.6350 at step 93).
```

## Installation

```bash
# Core (PyTorch only)
pip install traintools

# With curve fitting + plotting
pip install traintools[full]

# With HuggingFace Trainer integration
pip install traintools[hf]
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- scipy, numpy, matplotlib (optional, for `[full]`)
- transformers >= 4.30 (optional, for `[hf]`)

## License

MIT
