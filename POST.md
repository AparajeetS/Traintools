# Gradient Noise Scale for free, during gradient accumulation

If you train with gradient accumulation, you're already computing everything you
need to know whether your batch size is right — and throwing it away.

The **Gradient Noise Scale** (GNS, McCandlish et al. 2018) tells you the critical
batch size: the point past which bigger batches stop helping. Every existing
implementation estimates it with *extra* forward/backward passes. But the
per-micro-batch gradients you compute during accumulation are exactly the samples
GNS needs. So you can get it for free:

```python
from traintools import GradientAccumulationGNS

gns = GradientAccumulationGNS(model, micro_batch_size=B_micro)

for step in range(num_steps):
    for micro in micro_batches:
        (loss_fn(model(xm), ym) / accum_steps).backward()
        gns.record_microbatch()          # <- the only added line
    optimizer.step()
    result = gns.compute(step=step)      # GNS, zero extra passes
    optimizer.zero_grad()
    gns.reset_accumulation()
```

```
pip install traintools[full]
```

This is part of `traintools` — three training diagnostics with a 2-line API.
I'll be honest up front about what's novel and what isn't.

---

## 1. Gradient Noise Scale — is your batch size right?

```
GNS = tr(Σ) / ||G||²
```

tr(Σ) is the total variance of the per-example gradients; ‖G‖² is the squared
true gradient. Their ratio is the critical batch size B*.

- `GNS > B` → **under-batched**: the gradient is too noisy for this batch; bigger batches improve every step.
- `GNS < B` → **over-batched**: you're averaging more than you need to; shrink the batch and save compute.
- `GNS ≈ B` → **optimal**.

**The estimator matters.** The naive single-shot estimate is biased low — by
about 2× at the common 2-split setting — and far too noisy to act on. `traintools`
uses the paper's unbiased estimators (Bessel-corrected variance, bias-corrected
signal) and tracks GNS as the ratio of two separate exponential moving averages.
On a synthetic problem with a known true GNS of 4.0, the corrected estimator
recovers 3.99; the naive version returns 1.78.

---

## 2. PlasticityProbe — is your network becoming brittle?

Networks lose the ability to learn over long runs and repeated fine-tuning
(Dohare & Sutton, *Nature* 2024). The damage is in the *representations*, so
`traintools` measures activations directly:

- **Dormant unit fraction** — units that output ~0 for every input, attributed
  to the activation module that produced them.
- **Feature effective rank** — the effective rank of the activation covariance,
  normalised to [0,1]. Low rank means the representation has collapsed.

Combined into a **Plasticity Score ∈ [0,1]**:

```
[step 4000] Plasticity Score: 0.21
  Critical layers: blocks.11.mlp.act, blocks.10.mlp.act
  Action: reinitialise dormant units (continual backprop) or add regenerative regularisation.
```

---

## 3. TrainGuard — should you stop training yet?

Fits a power-law (or exponential) curve to the validation-loss history,
bootstraps uncertainty over the fit, and issues a STOP only when the 90%
confidence interval on further improvement falls below your threshold:

```
[step 193] STOP
  current loss: 0.6536
  predicted final: 0.6119
  expected improvement: 0.0417 (90% CI: [0.0012, 0.0821])
  reason: No improvement in 300 steps (best=0.6350 at step 93).
```

It also refuses to fire when the fit is unstable (predicted improvement < 0),
so a noisy curve doesn't get a false stop signal.

---

## Demo: the corrected GNS finds something the buggy version hid

I trained a small MLP on 2000 MNIST samples with 30% label noise — a setup with
genuinely high gradient noise.

![traintools demo](traintools_demo.png)

**GNS (left):** the corrected estimator reports the run is *severely
under-batched* — the critical batch is in the thousands, while training ran at
batch 64. Early on the signal is below the noise floor (GNS pinned at the display
cap); as training sharpens the gradient, GNS settles toward ~3000. That's a real,
actionable finding: on noisy data, much larger batches would denoise each step.

A buggy first version of this tool reported GNS ≈ 34 ("optimal") on the same run.
It was wrong by two orders of magnitude. The fix — Bessel correction plus a
bias-corrected signal — is the difference between a useless number and a useful
one.

**Plasticity (center):** the feature-rank score rises from 0.61 to 0.71 as the
representation develops, then stabilises — healthy. (20 epochs on 2000 samples
isn't long enough to kill units, which is the correct read.)

**TrainGuard (right):** the bouncing val loss is fit to a power law, the asymptote
projected at ~0.70, and training stopped at epoch 19 instead of running the full
budget.

---

## What's novel, honestly

Each metric has a paper behind it — none of the three are new science:

| Tool | Prior art |
|---|---|
| GNS | McCandlish et al. 2018 |
| PlasticityProbe | Dohare & Sutton 2024; Lyle et al. 2023 |
| TrainGuard | learning-curve extrapolation (Domhan 2015) |

The contribution is **packaging and ergonomics**: a correct, EMA-stabilised GNS
estimator that drops into any loop, and — the one genuinely differentiated piece —
computing it for *free* during gradient accumulation instead of paying for extra
passes. If that saves you from running a sweep to find the right batch size, it
paid for itself.

---

## Install & integrate

```bash
pip install traintools[full]   # scipy + matplotlib for fitting and plots
pip install traintools[hf]     # HuggingFace Trainer callback
```

```python
# HuggingFace Trainer — one line
from traintools.callbacks.huggingface import TraintoolsCallback
trainer = Trainer(model=model, ..., callbacks=[TraintoolsCallback()])
```

Source: https://github.com/AparajeetS/Traintools
PyPI: https://pypi.org/project/traintools/

Bug reports and PRs welcome — especially benchmarks of the free-accumulation GNS
against the extra-pass estimate on real LLM fine-tuning runs.

---

*References*
McCandlish, Kaplan, Amodei et al. (2018). An Empirical Model of Large-Batch Training. arXiv:1812.06162
Dohare, Sutton et al. (2024). Loss of plasticity in deep continual learning. Nature 632, 768–774.
Lyle et al. (2023). Understanding plasticity in neural networks. ICML.
Domhan et al. (2015). Speeding up automatic hyperparameter optimization of DNNs by extrapolation of learning curves. IJCAI.
