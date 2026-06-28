"""
Gradient confusion diagnostics.

Sankararaman et al. 2019 introduced gradient confusion as a way to reason about
when stochastic gradients from different samples fight each other and slow SGD.
This module exposes a small operational proxy: pairwise cosine similarity between
micro-batch gradients. Negative cosines indicate conflicting gradient directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class GradientConfusionResult:
    step: int
    n_gradients: int
    mean_cosine: float
    min_cosine: float
    max_cosine: float
    negative_fraction: float
    confusion_score: float
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def __str__(self) -> str:
        status = "OK" if self.ok else "WARN"
        lines = [
            f"[step {self.step}] GradientConfusion: {status} "
            f"mean_cos={self.mean_cosine:.3f} min_cos={self.min_cosine:.3f} "
            f"negative={self.negative_fraction:.1%} score={self.confusion_score:.3f}"
        ]
        for warning in self.warnings[:6]:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


def gradient_confusion_from_grads(
    micro_grads: torch.Tensor,
    *,
    step: int = 0,
    warn_negative_fraction: float = 0.25,
    warn_min_cosine: float = -0.2,
) -> GradientConfusionResult:
    """
    Compute pairwise gradient-conflict statistics from stacked gradients.

    Args:
        micro_grads: Tensor shaped (n_microbatches, n_parameters).
    """
    if micro_grads.dim() != 2:
        raise ValueError("micro_grads must have shape (n_gradients, n_parameters)")
    n = int(micro_grads.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 gradients to estimate confusion")

    g = micro_grads.detach().float()
    norms = g.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    g = g / norms
    cosine = g @ g.t()
    mask = ~torch.eye(n, dtype=torch.bool, device=cosine.device)
    pairs = cosine[mask]
    mean_cosine = float(pairs.mean().item())
    min_cosine = float(pairs.min().item())
    max_cosine = float(pairs.max().item())
    negative_fraction = float((pairs < 0).float().mean().item())
    confusion_score = max(0.0, -min_cosine) * negative_fraction

    warnings: List[str] = []
    if negative_fraction >= warn_negative_fraction:
        warnings.append(f"{negative_fraction:.1%} of gradient pairs are conflicting")
    if min_cosine <= warn_min_cosine:
        warnings.append(f"strongly opposed gradients detected (min cosine {min_cosine:.3f})")

    return GradientConfusionResult(
        step=step,
        n_gradients=n,
        mean_cosine=mean_cosine,
        min_cosine=min_cosine,
        max_cosine=max_cosine,
        negative_fraction=negative_fraction,
        confusion_score=confusion_score,
        warnings=warnings,
    )


class GradientConfusionMonitor:
    """
    Estimate gradient confusion by splitting a batch into micro-batches.

    This costs `n_splits` extra forward/backward passes. For a zero-extra-pass
    path, collect micro-batch gradients during gradient accumulation and call
    `gradient_confusion_from_grads(...)` directly.
    """

    def __init__(
        self,
        *,
        n_splits: int = 4,
        warn_negative_fraction: float = 0.25,
        warn_min_cosine: float = -0.2,
        eval_mode: bool = True,
    ) -> None:
        self.n_splits = n_splits
        self.warn_negative_fraction = warn_negative_fraction
        self.warn_min_cosine = warn_min_cosine
        self.eval_mode = eval_mode

    def estimate(
        self,
        model: nn.Module,
        loss_fn,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        step: int = 0,
    ) -> GradientConfusionResult:
        batch = int(inputs.shape[0])
        if batch < 2 * self.n_splits:
            raise ValueError(f"Batch size {batch} must be >= 2 * n_splits ({2 * self.n_splits})")
        micro = batch // self.n_splits
        grads = []
        saved_grads = {
            name: (p.grad.clone() if p.grad is not None else None)
            for name, p in model.named_parameters()
        }
        was_training = model.training
        if self.eval_mode:
            model.eval()

        try:
            for i in range(self.n_splits):
                sl = slice(i * micro, (i + 1) * micro)
                model.zero_grad(set_to_none=True)
                loss = loss_fn(model(inputs[sl]), targets[sl])
                loss.backward()
                parts = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
                if not parts:
                    raise RuntimeError("No gradients found during confusion estimate")
                grads.append(torch.cat(parts).clone())
        finally:
            model.zero_grad(set_to_none=True)
            for name, p in model.named_parameters():
                if saved_grads[name] is not None:
                    p.grad = saved_grads[name]
            if was_training:
                model.train()

        return gradient_confusion_from_grads(
            torch.stack(grads),
            step=step,
            warn_negative_fraction=self.warn_negative_fraction,
            warn_min_cosine=self.warn_min_cosine,
        )
