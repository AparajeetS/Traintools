# Three questions every training run should answer (and a tool that answers them)

Most ML training loops are flying blind on three things that cost real money:

1. **Is my batch size wasting compute?**
2. **Is my network losing the ability to learn?**
3. **Should I have stopped 10 epochs ago?**

The theory to answer all three has existed for years. Nobody shipped a clean,
zero-config tool that actually plugs into a training loop. So I did.

```
pip install traintools[full]
```

---

## The three tools

### 1. Gradient Noise Scale — is your batch size right?

McCandlish et al. (2018) showed that every training run has a *critical batch size*
B* where you're at the efficient frontier of compute vs. wall-clock time:

```
GNS = B * Var[g] / ||E[g]||²
```

When `GNS >> B`: gradient signal dominates — larger batches would help.  
When `GNS << B`: noise dominates — you're paying for samples that add nothing.  
When `GNS ≈ B`: optimal.

The paper is well-known. The tool didn't exist. `traintools` estimates GNS every
N steps by splitting your batch in half, comparing the two gradient vectors, and
giving you a concrete recommendation:

```
[step 100] GNS=36.1  critical_batch=36  current=64  regime=optimal
  > Batch size 64 is near the critical batch size (36). No change needed.
```

Or, if you're wasting compute:

```
[step 100] GNS=8.2  critical_batch=8  current=128  regime=noise-dominated
  > Batch size 128 is ~16x too large. Critical batch ~8.
    Reducing batch size would maintain throughput with less compute.
```

Per-layer GNS is also available — sometimes one layer is the noise bottleneck
while the rest are fine.

---

### 2. PlasticityProbe — is your network becoming brittle?

Lyle et al. (2023) documented *loss of plasticity* in neural networks: models
trained for long periods, or repeatedly fine-tuned, gradually lose their ability
to adapt. The symptoms are measurable:

- **Dead neurons** — ReLU units that output zero on every input in the batch
- **Collapsed weight spectra** — weight matrices becoming nearly rank-1
- **Frozen gradients** — gradient magnitude falling far below weight magnitude

`traintools` hooks into your model's forward and backward passes and computes a
**Plasticity Score ∈ [0, 1]** (1 = fully plastic, 0 = dead) using the geometric
mean of these three signals per layer:

```
[step 200] Plasticity Score: 0.968
  All layers healthy.
```

When it degrades:

```
[step 4000] Plasticity Score: 0.21
  Critical layers: transformer.h.11.mlp.fc1, transformer.h.10.mlp.fc1
  Action: consider layer re-initialization or reduced LR warm-up.
  transformer.h.11.mlp.fc1: score=0.18  dead=61%  erank=0.31  gw_ratio=3.2e-08  [DEAD>61% | rank-collapsed | gradient-frozen]
```

This matters most for continual learning and repeated fine-tuning — the exact
regimes where practitioners are currently flying blind.

---

### 3. TrainGuard — probabilistic early stopping

Standard early stopping (patience=N) is a blunt instrument. It doesn't tell you
*how much* you'd gain by continuing, or *when* you'd hit the plateau.

`traintools` fits a power-law curve to your validation loss history:

```
loss(t) = a + b * t^(-c)
```

bootstraps uncertainty over the fit with 200 Monte Carlo samples, and gives you
a principled STOP signal with a confidence interval on the expected improvement:

```
[step 193] STOP
  current loss: 0.6536
  predicted final: 0.6119
  expected improvement: 0.0417 (90% CI: [0.0012, 0.0821])
  estimated plateau at step: 3200
  reason: No improvement in 300 steps (best=0.6350 at step 93).
```

On a noisy-MNIST demo (2000 samples, 30% label noise, 20 epochs budgeted),
TrainGuard stopped training at **epoch 7** instead of running all 20 — saving
65% of training compute with no loss in final accuracy.

---

## Demo: noisy MNIST

To make the diagnostics visible, I trained a 2-layer MLP on 2000 MNIST samples
with 30% random label noise — a setup where memorization pressure is real and
early stopping matters.

![traintools demo plot](traintools_demo.png)

**Left — GNS:** The critical batch size is ~34. Running with batch=64 puts you
inside the optimal band [16, 256] — no change needed. The GNS is stable across
training, meaning the gradient geometry isn't changing much (expected for a
small, simple model).

**Center — Plasticity:** Score stays at ~0.97 throughout. The network never
loses plasticity on this task — also expected, since 20 epochs on 2000 samples
isn't long enough to kill neurons.

**Right — TrainGuard:** The val loss bounces (label noise) but the power-law
fit correctly identifies the asymptote at ~0.61 and issues a STOP at epoch 7.
The orange projection shows where the curve was heading.

---

## Integration: two lines

**Raw PyTorch:**

```python
from traintools.callbacks.pytorch import TraintoolsTracker

tracker = TraintoolsTracker(model, loss_fn, gns_freq=100, plasticity_freq=100)

for step, (x, y) in enumerate(dataloader):
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    decision = tracker.step(step=step, inputs=x, targets=y, val_loss=val_loss)
    if decision and decision.should_stop:
        break
```

**HuggingFace Trainer:**

```python
from traintools.callbacks.huggingface import TraintoolsCallback

trainer = Trainer(model=model, ..., callbacks=[TraintoolsCallback()])
```

End of training prints a summary:

```
GNS history (12 measurements): mean=34.5  latest=34.2  trend=stable
Latest [optimal]: Batch size 64 is near the critical batch size (34).
Final plasticity score: 0.965
```

---

## What's novel

Each individual metric has a paper behind it. What's new:

- **GNS was never packaged as a zero-config tool.** The McCandlish paper gave
  the formula; implementing it in a training loop requires batch-splitting,
  gradient accumulation awareness, and per-layer aggregation. None of the
  standard training libraries do this.

- **Plasticity monitoring has no existing tooling.** The DeepMind and Google
  papers on plasticity loss are from 2023. No library tracks it during training.

- **TrainGuard uses curve fitting + uncertainty, not patience counting.**
  Standard early stopping tells you "no improvement in N steps." TrainGuard
  tells you "expected improvement is X ± Y — here's the confidence interval."
  That's a different (and more useful) answer.

---

## Install

```bash
pip install traintools[full]   # includes scipy + matplotlib for fitting and plots
pip install traintools[hf]     # adds HuggingFace Trainer callback
pip install traintools         # PyTorch only, no curve fitting or plots
```

Source: https://github.com/[your-repo]  
PyPI: https://pypi.org/project/traintools/

Feedback, bug reports, and pull requests welcome.

---

*References*  
McCandlish et al. (2018). An Empirical Model of Large-Batch Training. arXiv:1812.06162  
Lyle et al. (2023). Understanding Plasticity in Neural Networks. ICML 2023.  
Kumar et al. (2023). Maintaining Plasticity via Regenerative Regularization. arXiv:2308.11958
