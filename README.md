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

GNS measures the ratio of gradient signal to noise across your batch:

```
GNS = B * Var[g] / ||E[g]||^2
```

When `GNS >> B` your batch is too small — gradient quality improves with more data.  
When `GNS << B` your batch is too large — you're wasting compute.  
When `GNS ≈ B` you're at the efficient frontier.

Output:

```
[step 100] GNS=36.1  critical_batch=36  current=64  regime=optimal
  > Batch size 64 is near the critical batch size (36). No change needed.
```

Reference: McCandlish et al. 2018, *An Empirical Model of Large-Batch Training*.

## PlasticityProbe

Networks lose plasticity over long training runs or repeated fine-tuning. PlasticityProbe
tracks three signals per layer:

- **Dead neuron fraction** — neurons that never activate on any batch sample
- **Effective rank** — how collapsed the weight matrix spectrum is
- **Gradient/weight ratio** — whether a layer is still receiving meaningful updates

Combined into a **Plasticity Score ∈ [0, 1]** (1 = fully plastic, 0 = dead).

```
[step 200] Plasticity Score: 0.968
  All layers healthy.
```

Reference: Lyle et al. 2023, *Understanding Plasticity in Neural Networks*.

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
