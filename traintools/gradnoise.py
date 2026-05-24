"""
Gradient Noise Scale (GNS) tracker.

GNS = tr(Σ) / ||G||²

where G is the true (full-dataset) gradient and Σ is the covariance of the
per-example gradients. At the critical batch size GNS ~ B.

  GNS >> B  → signal-dominated: larger batches improve gradient quality
  GNS << B  → noise-dominated:  smaller batches waste no compute
  GNS ~  B  → optimal: you're near the efficient frontier

This implementation uses the *unbiased* estimators from McCandlish et al. 2018:
given n micro-batch gradients g_1..g_n (each averaged over m examples),

  V        = Σ_i ||g_i − ḡ||² / (n − 1)      # Bessel-corrected variance of micro-grads
  tr(Σ)    = m · V                            # unbiased per-example gradient variance
  ||G||²   = ||ḡ||² − V / n                   # unbiased signal (removes residual noise)
  GNS      = tr(Σ) / ||G||²

Per-call estimates are noisy, so GNS is tracked as the ratio of two separate
exponential moving averages (EMA of tr(Σ)) / (EMA of ||G||²) — exactly as
recommended in the paper. This is far more stable than averaging GNS itself.

Two ways to obtain the micro-batch gradients:
  1. estimate_gns(...)        — extra forward/backward passes on a held batch
                                (forces eval() to exclude dropout/BN noise).
  2. GradientAccumulationGNS  — FREE: reuses the micro-batch gradients you
                                already compute during gradient accumulation.

Reference: McCandlish, Kaplan, Amodei et al. 2018,
           "An Empirical Model of Large-Batch Training", arXiv:1812.06162.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class GNSResult:
    step: int
    gns: float                          # gradient noise scale (EMA-based if available)
    critical_batch: int                 # round(GNS) — recommended batch size
    current_batch: int
    regime: str                         # 'noise-dominated' | 'optimal' | 'signal-dominated'
    recommendation: str
    tr_sigma: float = 0.0               # estimated tr(Σ)  (gradient variance trace)
    g_squared: float = 0.0              # estimated ||G||²  (signal)
    raw_gns: float = 0.0                # single-shot estimate (no EMA)
    smoothed: bool = False              # whether `gns` is EMA-smoothed
    per_layer: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        tag = "EMA" if self.smoothed else "raw"
        lines = [
            f"[step {self.step}] GNS={self.gns:.1f} ({tag})  critical_batch={self.critical_batch}  "
            f"current={self.current_batch}  regime={self.regime}",
            f"  > {self.recommendation}",
        ]
        if self.per_layer:
            noisy = sorted(self.per_layer.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("  noisy layers: " + ", ".join(f"{n}={v:.1f}" for n, v in noisy))
        return "\n".join(lines)


# ── Core unbiased estimator ──────────────────────────────────────────────────

def _unbiased_estimates(micro_grads: torch.Tensor, micro_size: int) -> tuple[float, float]:
    """
    Given micro-batch gradients stacked as (n_splits, D), each an average over
    `micro_size` examples, return unbiased (tr_sigma, g_squared).

    tr_sigma  = micro_size * V          where V is Bessel-corrected variance
    g_squared = ||ḡ||² − V / n_splits   (removes residual noise from the signal)
    """
    n = micro_grads.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 micro-batch gradients to estimate GNS variance.")
    g_mean = micro_grads.mean(0)
    V = ((micro_grads - g_mean) ** 2).sum() / (n - 1)   # Bessel-corrected
    tr_sigma = float(micro_size * V)
    g_squared = float((g_mean ** 2).sum() - V / n)      # unbiased signal
    return tr_sigma, g_squared


# Cap GNS so a near-zero (noise-swamped) signal yields a large finite number
# instead of inf, which is useless to display and act on.
GNS_MAX = 1.0e5


def _gns_from_estimates(tr_sigma: float, g_squared: float) -> float:
    """
    GNS = tr(Σ)/||G||².

    When the unbiased signal is non-positive, the gradient noise overwhelms any
    measurable signal — the batch is far below the critical size. We report a
    large finite cap rather than inf so the value is plottable and comparable.
    """
    if g_squared <= 0 or tr_sigma / g_squared > GNS_MAX:
        return GNS_MAX
    return tr_sigma / g_squared


def _classify(gns: float, B: int) -> tuple[str, str]:
    """
    Regime naming is from the practitioner's point of view:
      under-batched  (GNS > 4B):   gradient too noisy for this batch — increase B
      over-batched   (GNS < 0.25B): batch larger than needed — decrease B (save compute)
      optimal        otherwise
    """
    capped = gns >= GNS_MAX
    critical = min(max(1, round(gns)), 100 * B)  # cap displayed critical batch
    if gns > 4.0 * B:
        regime = "under-batched"
        if capped:
            rec = (f"Gradient noise far exceeds signal - batch {B} is much too small. "
                   f"Critical batch is at least ~{critical}; increase batch size substantially "
                   f"(this dataset/stage is very noisy).")
        else:
            rec = (f"Batch size {B} is ~{max(1, critical // B)}x below the critical batch (~{critical}). "
                   f"Larger batches would give cleaner gradients per step.")
    elif gns < 0.25 * B:
        regime = "over-batched"
        factor = max(1, B // max(1, critical))
        rec = (f"Batch size {B} is ~{factor}x larger than the critical batch (~{critical}). "
               f"Reducing batch size (and scaling LR) would cut compute with little quality loss.")
    else:
        regime = "optimal"
        rec = (f"Batch size {B} is near the critical batch size (~{critical}). "
               f"Gradient signal is well-conditioned - no change needed.")
    return regime, rec


# ── Stateless extra-pass estimator ───────────────────────────────────────────

def _flat_grads(model: nn.Module) -> torch.Tensor:
    parts = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    if not parts:
        raise RuntimeError("No gradients found. Call backward() before estimating GNS.")
    return torch.cat(parts)


def estimate_gns(
    model: nn.Module,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_splits: int = 2,
    step: int = 0,
    per_layer: bool = False,
    eval_mode: bool = True,
) -> GNSResult:
    """
    Estimate GNS via `n_splits` extra forward/backward passes on the given batch.

    By default the model is switched to eval() during estimation so that dropout
    and BatchNorm do not inject non-data noise into the gradient variance. The
    caller's gradients and train/eval state are restored exactly on return.

    Cost: n_splits forward+backward passes. Call every 100–500 steps.

    For zero-overhead estimation during gradient accumulation, use
    GradientAccumulationGNS instead.
    """
    B = inputs.size(0)
    if B < 2 * n_splits:
        raise ValueError(f"Batch size {B} must be >= 2 * n_splits ({2 * n_splits})")

    micro_size = B // n_splits
    micro_grads: List[torch.Tensor] = []
    micro_layer: List[Dict[str, torch.Tensor]] = []

    # Snapshot caller's gradients + training state
    saved_grads = {n: (p.grad.clone() if p.grad is not None else None)
                   for n, p in model.named_parameters()}
    was_training = model.training
    if eval_mode:
        model.eval()

    try:
        for i in range(n_splits):
            sl = slice(i * micro_size, (i + 1) * micro_size)
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs[sl]), targets[sl])
            loss.backward()
            micro_grads.append(_flat_grads(model))
            if per_layer:
                micro_layer.append({n: p.grad.detach().reshape(-1).clone()
                                    for n, p in model.named_parameters() if p.grad is not None})
    finally:
        model.zero_grad(set_to_none=True)
        for n, p in model.named_parameters():
            if saved_grads[n] is not None:
                p.grad = saved_grads[n]
        if was_training:
            model.train()

    g = torch.stack(micro_grads)
    tr_sigma, g_sq = _unbiased_estimates(g, micro_size)
    gns = _gns_from_estimates(tr_sigma, g_sq)

    layer_gns: Dict[str, float] = {}
    if per_layer:
        for name in micro_layer[0]:
            gl = torch.stack([d[name] for d in micro_layer])
            t_l, s_l = _unbiased_estimates(gl, micro_size)
            layer_gns[name] = _gns_from_estimates(t_l, s_l)

    regime, rec = _classify(gns, B)
    return GNSResult(
        step=step, gns=gns, critical_batch=min(max(1, round(gns)), 100 * B),
        current_batch=B, regime=regime, recommendation=rec,
        tr_sigma=tr_sigma, g_squared=g_sq, raw_gns=gns, smoothed=False, per_layer=layer_gns,
    )


# ── Stateful EMA estimator (McCandlish-recommended) ──────────────────────────

class GNSEstimator:
    """
    Maintains separate exponential moving averages of tr(Σ) and ||G||², and
    reports GNS = EMA(trΣ) / EMA(||G||²). This is the stable estimator the
    paper recommends — single-shot GNS is too noisy to act on.

    Feed it micro-batch gradients (stacked (n, D)) each step via `update`.
    """

    def __init__(self, decay: float = 0.95) -> None:
        self.decay = decay
        self._ema_tr: Optional[float] = None
        self._ema_sig: Optional[float] = None

    def reset(self) -> None:
        self._ema_tr = None
        self._ema_sig = None

    def update(self, micro_grads: torch.Tensor, micro_size: int,
               current_batch: int, step: int = 0,
               per_layer_grads: Optional[Dict[str, torch.Tensor]] = None) -> GNSResult:
        tr_sigma, g_sq = _unbiased_estimates(micro_grads, micro_size)
        layer_gns: Dict[str, float] = {}
        if per_layer_grads is not None:
            for name, gl in per_layer_grads.items():
                t_l, s_l = _unbiased_estimates(gl, micro_size)
                layer_gns[name] = _gns_from_estimates(t_l, s_l)
        return self.update_from_estimates(tr_sigma, g_sq, current_batch,
                                          step=step, per_layer=layer_gns)

    def update_from_estimates(self, tr_sigma: float, g_sq: float, current_batch: int,
                              step: int = 0,
                              per_layer: Optional[Dict[str, float]] = None) -> GNSResult:
        """Update the EMAs directly from precomputed (tr_sigma, g_sq) estimates."""
        raw_gns = _gns_from_estimates(tr_sigma, g_sq)
        b = self.decay
        self._ema_tr = tr_sigma if self._ema_tr is None else b * self._ema_tr + (1 - b) * tr_sigma
        self._ema_sig = g_sq if self._ema_sig is None else b * self._ema_sig + (1 - b) * g_sq
        gns = _gns_from_estimates(self._ema_tr, self._ema_sig)
        regime, rec = _classify(gns, current_batch)
        return GNSResult(
            step=step, gns=gns,
            critical_batch=min(max(1, round(gns)), 100 * current_batch),
            current_batch=current_batch, regime=regime, recommendation=rec,
            tr_sigma=self._ema_tr, g_squared=self._ema_sig, raw_gns=raw_gns,
            smoothed=True, per_layer=per_layer or {},
        )


# ── Free GNS during gradient accumulation ────────────────────────────────────

class GradientAccumulationGNS:
    """
    Compute GNS for FREE during gradient accumulation.

    Standard accumulation already computes a gradient for each micro-batch and
    sums them into .grad. Those per-micro-batch gradients are exactly the
    samples GNS needs — so we recover them by differencing the running .grad
    after each micro-batch backward, at zero extra forward/backward cost.

    Usage:
        gns = GradientAccumulationGNS(model, micro_batch_size=B_micro)

        for step in range(num_steps):
            for micro in range(accum_steps):
                loss = loss_fn(model(x_micro), y_micro) / accum_steps
                loss.backward()
                gns.record_microbatch()          # <-- after each backward
            optimizer.step()
            result = gns.compute(step=step)      # GNSResult (or None if <2 micro-batches)
            optimizer.zero_grad()
            gns.reset_accumulation()

    Notes:
      * Works regardless of whether you scale the loss by 1/accum_steps — GNS is
        invariant to a global gradient rescaling.
      * Captures the *real* training-time gradient noise (including dropout/BN),
        which is what you actually optimize against. Use estimate_gns(eval_mode=True)
        if you instead want pure data-sampling noise.
      * Requires accum_steps >= 2 to estimate variance.
    """

    def __init__(self, model: nn.Module, micro_batch_size: int, decay: float = 0.95) -> None:
        self.model = model
        self.micro_batch_size = micro_batch_size
        self.estimator = GNSEstimator(decay=decay)
        self._prev_cumulative: Optional[torch.Tensor] = None
        self._micro_grads: List[torch.Tensor] = []

    def _current_flat_grad(self) -> torch.Tensor:
        parts = [p.grad.detach().reshape(-1) for p in self.model.parameters() if p.grad is not None]
        return torch.cat(parts) if parts else torch.empty(0)

    def record_microbatch(self) -> None:
        """Call once after each micro-batch's backward(), before optimizer.zero_grad()."""
        cur = self._current_flat_grad().clone()
        if cur.numel() == 0:
            return
        if self._prev_cumulative is None:
            micro = cur
        else:
            micro = cur - self._prev_cumulative
        self._micro_grads.append(micro)
        self._prev_cumulative = cur

    def compute(self, step: int = 0) -> Optional[GNSResult]:
        """Compute the GNS estimate from this step's recorded micro-batches."""
        if len(self._micro_grads) < 2:
            return None
        g = torch.stack(self._micro_grads)
        full_batch = self.micro_batch_size * len(self._micro_grads)
        return self.estimator.update(g, self.micro_batch_size, full_batch, step=step)

    def reset_accumulation(self) -> None:
        """Clear per-step buffers. Call after compute(), once per optimizer step."""
        self._prev_cumulative = None
        self._micro_grads = []


# ── History / trend tracking ─────────────────────────────────────────────────

class GNSHistory:
    """Running log of GNS estimates across training, with trend analysis."""

    def __init__(self) -> None:
        self.results: List[GNSResult] = []

    def record(self, result: GNSResult) -> None:
        self.results.append(result)

    @property
    def steps(self) -> List[int]:
        return [r.step for r in self.results]

    @property
    def values(self) -> List[float]:
        return [r.gns for r in self.results]

    def trend(self) -> str:
        vs = [v for v in self.values[-5:] if math.isfinite(v)]
        if len(vs) < 2:
            return "insufficient data"
        slope = (vs[-1] - vs[0]) / max(1, len(vs) - 1)
        rel = abs(slope) / (sum(vs) / len(vs) + 1e-12)
        if rel < 0.05:
            return "stable"
        return "rising" if slope > 0 else "falling"

    def summary(self) -> str:
        if not self.results:
            return "No GNS measurements yet."
        finite = [v for v in self.values if math.isfinite(v)]
        latest = self.results[-1]
        mean_str = f"{sum(finite)/len(finite):.1f}" if finite else "n/a"
        return (
            f"GNS history ({len(self.values)} measurements): "
            f"mean={mean_str}  latest={latest.gns:.1f}  trend={self.trend()}\n"
            f"Latest [{latest.regime}]: {latest.recommendation}"
        )

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))
        finite_steps = [s for s, v in zip(self.steps, self.values) if math.isfinite(v)]
        finite_vals = [v for v in self.values if math.isfinite(v)]
        ax.plot(finite_steps, finite_vals, marker="o", linewidth=1.5, label="GNS (EMA)")
        if self.results:
            B = self.results[-1].current_batch
            ax.axhspan(0.25 * B, 4.0 * B, alpha=0.12, color="green", label="optimal band")
            ax.axhline(B, color="green", linestyle="--", linewidth=0.8, label=f"current batch={B}")
        # GNS spans orders of magnitude — log scale is far more readable.
        if finite_vals and max(finite_vals) / (min(finite_vals) + 1e-9) > 15:
            ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("GNS")
        ax.set_title("Gradient Noise Scale")
        ax.legend(fontsize=8)
        return ax
