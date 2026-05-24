"""
Gradient Noise Scale (GNS) tracker.

GNS = B * Var[g] / ||E[g]||^2

At the critical batch size GNS ≈ B.
  GNS >> B  → signal-dominated: larger batches improve gradient quality
  GNS << B  → noise-dominated:  smaller batches waste no compute
  GNS ≈  B  → optimal: you're near the efficient frontier

Reference: McCandlish et al. 2018 "An Empirical Model of Large-Batch Training"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class GNSResult:
    step: int
    gns: float                          # global gradient noise scale
    critical_batch: int                 # round(GNS) — recommended batch size
    current_batch: int
    regime: str                         # 'noise-dominated' | 'optimal' | 'signal-dominated'
    recommendation: str
    per_layer: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"[step {self.step}] GNS={self.gns:.1f}  critical_batch={self.critical_batch}  "
            f"current={self.current_batch}  regime={self.regime}",
            f"  > {self.recommendation}",
        ]
        if self.per_layer:
            noisy = sorted(self.per_layer.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("  noisy layers: " + ", ".join(f"{n}={v:.1f}" for n, v in noisy))
        return "\n".join(lines)


def _flat_grads(model: nn.Module) -> torch.Tensor:
    parts = [p.grad.detach().clone().flatten() for p in model.parameters() if p.grad is not None]
    if not parts:
        raise RuntimeError("No gradients found. Call backward() before estimating GNS.")
    return torch.cat(parts)


def _per_layer_grads(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.grad.detach().clone().flatten()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def estimate_gns(
    model: nn.Module,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_splits: int = 2,
    step: int = 0,
    per_layer: bool = False,
) -> GNSResult:
    """
    Estimate GNS by splitting the batch into n_splits equal micro-batches,
    computing gradients on each, then measuring signal vs. noise.

    Cost: n_splits forward+backward passes.
    Recommended call frequency: every 100–500 training steps.

    Args:
        model:    the model being trained (must be in train mode)
        loss_fn:  callable(output, target) → scalar loss
        inputs:   a full batch of inputs (shape [B, ...])
        targets:  a full batch of targets (shape [B, ...])
        n_splits: how many micro-batches to split into (2 is usually enough)
        step:     current training step (for logging)
        per_layer: whether to compute per-layer GNS (slightly more expensive)

    Returns:
        GNSResult with global GNS, regime classification, and recommendation.
    """
    B = inputs.size(0)
    if B < 2 * n_splits:
        raise ValueError(f"Batch size {B} must be >= 2 * n_splits ({2 * n_splits})")

    micro_size = B // n_splits
    micro_grads: list[torch.Tensor] = []
    micro_layer_grads: list[Dict[str, torch.Tensor]] = []

    original_grads = {name: (p.grad.clone() if p.grad is not None else None)
                      for name, p in model.named_parameters()}

    try:
        for i in range(n_splits):
            start, end = i * micro_size, (i + 1) * micro_size
            x_i = inputs[start:end]
            y_i = targets[start:end]

            model.zero_grad()
            loss = loss_fn(model(x_i), y_i)
            loss.backward()

            micro_grads.append(_flat_grads(model))
            if per_layer:
                micro_layer_grads.append(_per_layer_grads(model))
    finally:
        # Restore original gradients so the caller's optimizer step is unaffected
        model.zero_grad()
        for name, p in model.named_parameters():
            if original_grads[name] is not None:
                p.grad = original_grads[name]

    # Stack: shape (n_splits, D)
    g = torch.stack(micro_grads)
    g_mean = g.mean(0)
    noise = ((g - g_mean) ** 2).mean(0).sum()       # tr(Σ_g)
    signal = (g_mean ** 2).sum()                     # ||E[g]||^2

    gns = float(micro_size * noise / (signal + 1e-12))
    critical = max(1, round(gns))

    # Per-layer
    layer_gns: Dict[str, float] = {}
    if per_layer:
        all_names = list(micro_layer_grads[0].keys())
        for name in all_names:
            gl = torch.stack([d[name] for d in micro_layer_grads])
            gl_mean = gl.mean(0)
            nl = ((gl - gl_mean) ** 2).mean(0).sum()
            sl = (gl_mean ** 2).sum()
            layer_gns[name] = float(micro_size * nl / (sl + 1e-12))

    # Classify and recommend
    if gns < 0.25 * B:
        regime = "noise-dominated"
        rec = (f"Batch size {B} is ~{B // max(1, critical)}x too large. "
               f"Critical batch ≈ {critical}. "
               f"Reducing batch size (or increasing learning rate proportionally) would maintain speed with less compute.")
    elif gns > 4.0 * B:
        regime = "signal-dominated"
        rec = (f"Batch size {B} is ~{critical // max(1, B)}x too small. "
               f"Critical batch ≈ {critical}. "
               f"Larger batches would give cleaner gradients per step.")
    else:
        regime = "optimal"
        rec = (f"Batch size {B} is near the critical batch size ({critical}). "
               f"Gradient signal is well-conditioned. No change needed.")

    return GNSResult(
        step=step,
        gns=gns,
        critical_batch=critical,
        current_batch=B,
        regime=regime,
        recommendation=rec,
        per_layer=layer_gns,
    )


class GNSHistory:
    """Running log of GNS estimates across training, with trend analysis."""

    def __init__(self) -> None:
        self.results: list[GNSResult] = []

    def record(self, result: GNSResult) -> None:
        self.results.append(result)

    @property
    def steps(self) -> list[int]:
        return [r.step for r in self.results]

    @property
    def values(self) -> list[float]:
        return [r.gns for r in self.results]

    def trend(self) -> str:
        """Returns 'rising', 'falling', or 'stable' based on last 5 estimates."""
        vs = self.values[-5:]
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
        gns_vals = self.values
        latest = self.results[-1]
        return (
            f"GNS history ({len(gns_vals)} measurements): "
            f"mean={sum(gns_vals)/len(gns_vals):.1f}  "
            f"latest={gns_vals[-1]:.1f}  trend={self.trend()}\n"
            f"Latest [{latest.regime}]: {latest.recommendation}"
        )

    def plot(self, ax=None):
        """Plot GNS over training steps. Requires matplotlib."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self.steps, self.values, marker="o", linewidth=1.5, label="GNS")
        # Shade the optimal band (0.25*B to 4*B) if batch size is consistent
        if self.results:
            B = self.results[-1].current_batch
            ax.axhspan(0.25 * B, 4.0 * B, alpha=0.12, color="green", label="optimal band")
            ax.axhline(B, color="green", linestyle="--", linewidth=0.8, label=f"current batch={B}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("GNS")
        ax.set_title("Gradient Noise Scale")
        ax.legend(fontsize=8)
        return ax
