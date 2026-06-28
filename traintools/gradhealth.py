"""
GradientHealthMonitor - gradient and update-ratio diagnostics for PyTorch.

Call after loss.backward() and before optimizer.step(). It does not mutate the
model. If you pass the learning rate, it also estimates the SGD-style
update-to-weight ratio, a useful signal for too-large or too-small steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class LayerGradientStats:
    name: str
    grad_norm: float
    param_norm: float
    update_ratio: Optional[float]
    finite: bool
    nan_count: int
    inf_count: int
    max_abs_grad: float
    zero_fraction: float


@dataclass
class GradientHealthResult:
    step: int
    total_grad_norm: float
    total_param_norm: float
    global_update_ratio: Optional[float]
    max_abs_grad: float
    clip_coef: Optional[float]
    layers: List[LayerGradientStats] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def __str__(self) -> str:
        status = "OK" if self.ok else "WARN"
        ratio = "n/a" if self.global_update_ratio is None else f"{self.global_update_ratio:.2e}"
        lines = [
            f"[step {self.step}] GradientHealth: {status} "
            f"grad_norm={self.total_grad_norm:.3g} update_ratio={ratio}"
        ]
        if self.clip_coef is not None and self.clip_coef < 1.0:
            lines.append(f"  clip coef for requested max norm: {self.clip_coef:.3g}")
        for warning in self.warnings[:8]:
            lines.append(f"  ! {warning}")
        if len(self.warnings) > 8:
            lines.append(f"  ! ... {len(self.warnings) - 8} more warnings")
        return "\n".join(lines)


class GradientHealthMonitor:
    """
    Inspect gradients for non-finite values, vanishing/exploding norms, and
    update-to-weight ratios.

    Usage:
        monitor = GradientHealthMonitor(max_grad_norm=1.0)
        loss.backward()
        report = monitor.inspect(model, step=step, lr=optimizer.param_groups[0]["lr"])
    """

    def __init__(
        self,
        *,
        vanishing_threshold: float = 1.0e-12,
        exploding_threshold: float = 1.0e3,
        max_update_ratio: float = 1.0e-1,
        min_update_ratio: float = 1.0e-8,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.max_update_ratio = max_update_ratio
        self.min_update_ratio = min_update_ratio
        self.max_grad_norm = max_grad_norm

    def inspect(
        self,
        model: nn.Module,
        *,
        step: int = 0,
        lr: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
    ) -> GradientHealthResult:
        layers: List[LayerGradientStats] = []
        warnings: List[str] = []
        total_grad_sq = 0.0
        total_param_sq = 0.0
        max_abs_grad = 0.0
        seen_grad = False

        for name, param in model.named_parameters():
            param_norm = float(param.detach().float().norm().item())
            total_param_sq += param_norm ** 2
            if param.grad is None:
                continue
            seen_grad = True
            grad = param.grad.detach().float()
            finite_mask = torch.isfinite(grad)
            nan_count = int(torch.isnan(grad).sum().item())
            inf_count = int(torch.isinf(grad).sum().item())
            finite = bool(finite_mask.all().item())

            finite_grad = grad[finite_mask]
            if finite_grad.numel() == 0:
                grad_norm = float("inf")
                layer_max_abs = float("inf")
                zero_fraction = 0.0
            else:
                grad_norm = float(finite_grad.norm().item())
                layer_max_abs = float(finite_grad.abs().max().item())
                zero_fraction = float((finite_grad == 0).float().mean().item())

            total_grad_sq += grad_norm ** 2 if math.isfinite(grad_norm) else float("inf")
            max_abs_grad = max(max_abs_grad, layer_max_abs)
            update_ratio = None
            if lr is not None and param_norm > 0 and math.isfinite(grad_norm):
                update_ratio = abs(lr) * grad_norm / (param_norm + 1.0e-12)

            stat = LayerGradientStats(
                name=name,
                grad_norm=grad_norm,
                param_norm=param_norm,
                update_ratio=update_ratio,
                finite=finite,
                nan_count=nan_count,
                inf_count=inf_count,
                max_abs_grad=layer_max_abs,
                zero_fraction=zero_fraction,
            )
            layers.append(stat)

            if not finite:
                warnings.append(f"{name} has non-finite gradients (nan={nan_count}, inf={inf_count})")
            if math.isfinite(grad_norm) and grad_norm <= self.vanishing_threshold:
                warnings.append(f"{name} gradient appears vanished (norm={grad_norm:.3g})")
            if not math.isfinite(grad_norm) or grad_norm >= self.exploding_threshold:
                warnings.append(f"{name} gradient is very large (norm={grad_norm:.3g})")
            if update_ratio is not None and update_ratio >= self.max_update_ratio:
                warnings.append(f"{name} update/weight ratio is high ({update_ratio:.2e})")

        total_grad_norm = math.sqrt(total_grad_sq) if math.isfinite(total_grad_sq) else float("inf")
        total_param_norm = math.sqrt(total_param_sq)
        global_update_ratio = None
        if lr is not None and total_param_norm > 0 and math.isfinite(total_grad_norm):
            global_update_ratio = abs(lr) * total_grad_norm / (total_param_norm + 1.0e-12)
            if global_update_ratio <= self.min_update_ratio and seen_grad:
                warnings.append(f"global update/weight ratio is tiny ({global_update_ratio:.2e})")

        if not seen_grad:
            warnings.append("no gradients found; call backward() before inspecting")

        requested_max = max_grad_norm if max_grad_norm is not None else self.max_grad_norm
        clip_coef = None
        if requested_max is not None and total_grad_norm > 0:
            clip_coef = min(1.0, requested_max / (total_grad_norm + 1.0e-12))
            if clip_coef < 1.0:
                warnings.append(
                    f"global grad norm {total_grad_norm:.3g} exceeds clipping threshold {requested_max:.3g}"
                )

        return GradientHealthResult(
            step=step,
            total_grad_norm=total_grad_norm,
            total_param_norm=total_param_norm,
            global_update_ratio=global_update_ratio,
            max_abs_grad=max_abs_grad,
            clip_coef=clip_coef,
            layers=layers,
            warnings=warnings,
        )
