"""
Neural collapse diagnostics for classification features.

Papyan, Han, and Donoho (PNAS 2020) described neural collapse: in late training,
within-class feature variability shrinks, class means approach a simplex ETF,
and classifier weights align with class means. This module exposes compact
measurements for those phenomena from a feature batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class NeuralCollapseResult:
    n_classes: int
    n_samples: int
    nc1_within_to_between: float
    nc2_etf_deviation: float
    nc3_classifier_alignment: Optional[float]
    ncc_accuracy: float
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def __str__(self) -> str:
        status = "OK" if self.ok else "WARN"
        nc3 = "n/a" if self.nc3_classifier_alignment is None else f"{self.nc3_classifier_alignment:.3f}"
        lines = [
            f"NeuralCollapse: {status} classes={self.n_classes} samples={self.n_samples} "
            f"NC1={self.nc1_within_to_between:.3g} NC2={self.nc2_etf_deviation:.3g} "
            f"NC3_align={nc3} NCC_acc={self.ncc_accuracy:.3f}"
        ]
        for warning in self.warnings[:6]:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


class NeuralCollapseMonitor:
    """
    Measure neural-collapse geometry from penultimate features and labels.

    Args:
        nc1_warn: warn if within-class variability remains large relative to
            between-class variability.
        ncc_warn: warn if nearest-class-center accuracy is low.
    """

    def __init__(self, *, nc1_warn: float = 1.0, ncc_warn: float = 0.8) -> None:
        self.nc1_warn = nc1_warn
        self.ncc_warn = ncc_warn

    def measure(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        classifier_weight: Optional[torch.Tensor] = None,
    ) -> NeuralCollapseResult:
        if features.dim() != 2:
            raise ValueError("features must have shape (n_samples, n_features)")
        x = features.detach().float()
        y = labels.detach().long().reshape(-1)
        if y.numel() != x.shape[0]:
            raise ValueError("labels must contain one value per feature row")

        classes = torch.unique(y).sort().values
        if classes.numel() < 2:
            raise ValueError("Need at least two classes to measure neural collapse")

        means = []
        within_sq = 0.0
        for cls in classes:
            rows = x[y == cls]
            mu = rows.mean(dim=0)
            means.append(mu)
            within_sq += float(((rows - mu) ** 2).sum().item())

        M = torch.stack(means)
        global_mean = x.mean(dim=0, keepdim=True)
        centered_means = M - global_mean
        between_sq = float((centered_means ** 2).sum().item())
        nc1 = within_sq / (between_sq + 1.0e-12)

        normed_means = centered_means / centered_means.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        cos = normed_means @ normed_means.t()
        k = int(classes.numel())
        off_mask = ~torch.eye(k, dtype=torch.bool, device=cos.device)
        target = -1.0 / max(1, k - 1)
        nc2 = float((cos[off_mask] - target).abs().mean().item())

        distances = torch.cdist(x, M)
        nearest = classes[distances.argmin(dim=1)]
        ncc_accuracy = float((nearest == y).float().mean().item())

        nc3_alignment: Optional[float] = None
        if classifier_weight is not None:
            W = classifier_weight.detach().float()
            if W.shape[0] >= k and W.shape[1] == x.shape[1]:
                W = W[classes]
                W = W - W.mean(dim=0, keepdim=True)
                Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
                nc3_alignment = float((Wn * normed_means).sum(dim=1).mean().item())

        warnings: List[str] = []
        if nc1 > self.nc1_warn:
            warnings.append(f"NC1 is high ({nc1:.3g}); within-class variability has not collapsed")
        if ncc_accuracy < self.ncc_warn:
            warnings.append(f"NCC accuracy is low ({ncc_accuracy:.1%}); class means are weak prototypes")

        return NeuralCollapseResult(
            n_classes=k,
            n_samples=int(x.shape[0]),
            nc1_within_to_between=nc1,
            nc2_etf_deviation=nc2,
            nc3_classifier_alignment=nc3_alignment,
            ncc_accuracy=ncc_accuracy,
            warnings=warnings,
        )
