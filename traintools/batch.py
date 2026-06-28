"""
BatchInspector - lightweight data/batch health checks for PyTorch training.

The goal is not to replace dataset validation. It is a cheap per-batch sentinel
for problems that quietly ruin runs: NaNs, infinities, exploding input scales,
constant tensors, empty tensors, and severe class imbalance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass
class TensorStats:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    numel: int
    finite_fraction: float
    nan_count: int
    inf_count: int
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    abs_max: Optional[float]
    zero_fraction: Optional[float]

    @property
    def ok(self) -> bool:
        return self.nan_count == 0 and self.inf_count == 0


@dataclass
class BatchReport:
    step: int
    batch_size: Optional[int]
    tensors: List[TensorStats] = field(default_factory=list)
    label_distribution: Dict[int, int] = field(default_factory=dict)
    imbalance_ratio: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def __str__(self) -> str:
        status = "OK" if self.ok else "WARN"
        lines = [
            f"[step {self.step}] BatchInspector: {status} "
            f"batch_size={self.batch_size} tensors={len(self.tensors)}"
        ]
        if self.label_distribution:
            lines.append(f"  labels: {self.label_distribution}")
        for warning in self.warnings[:8]:
            lines.append(f"  ! {warning}")
        if len(self.warnings) > 8:
            lines.append(f"  ! ... {len(self.warnings) - 8} more warnings")
        return "\n".join(lines)


def _iter_tensors(obj: Any, prefix: str = "batch") -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            yield from _iter_tensors(value, f"{prefix}.{key}")
    elif isinstance(obj, (tuple, list)):
        for i, value in enumerate(obj):
            yield from _iter_tensors(value, f"{prefix}.{i}")


def _first_dim(obj: Any) -> Optional[int]:
    for _, tensor in _iter_tensors(obj):
        if tensor.dim() >= 1:
            return int(tensor.shape[0])
    return None


def _stats(name: str, tensor: torch.Tensor) -> TensorStats:
    numel = tensor.numel()
    dtype = str(tensor.dtype).replace("torch.", "")
    shape = tuple(int(x) for x in tensor.shape)
    if numel == 0:
        return TensorStats(
            name=name, shape=shape, dtype=dtype, numel=0,
            finite_fraction=1.0, nan_count=0, inf_count=0,
            mean=None, std=None, min=None, max=None, abs_max=None,
            zero_fraction=None,
        )

    with torch.no_grad():
        if tensor.is_floating_point() or tensor.is_complex():
            values = tensor.detach().float()
            finite = torch.isfinite(values)
            nan_count = int(torch.isnan(values).sum().item())
            inf_count = int(torch.isinf(values).sum().item())
            finite_fraction = float(finite.float().mean().item())
            finite_values = values[finite]
        else:
            values = tensor.detach()
            nan_count = 0
            inf_count = 0
            finite_fraction = 1.0
            finite_values = values.float().reshape(-1)

        if finite_values.numel() == 0:
            mean = std = min_value = max_value = abs_max = zero_fraction = None
        else:
            flat = finite_values.float().reshape(-1)
            mean = float(flat.mean().item())
            std = float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0
            min_value = float(flat.min().item())
            max_value = float(flat.max().item())
            abs_max = float(flat.abs().max().item())
            zero_fraction = float((flat == 0).float().mean().item())

    return TensorStats(
        name=name, shape=shape, dtype=dtype, numel=numel,
        finite_fraction=finite_fraction, nan_count=nan_count, inf_count=inf_count,
        mean=mean, std=std, min=min_value, max=max_value, abs_max=abs_max,
        zero_fraction=zero_fraction,
    )


def _label_distribution(targets: Optional[torch.Tensor]) -> Dict[int, int]:
    if targets is None or not isinstance(targets, torch.Tensor) or targets.numel() == 0:
        return {}
    if targets.dim() == 0:
        labels = targets.reshape(1)
    elif targets.dim() == 1:
        labels = targets
    elif targets.is_floating_point() and targets.dim() >= 2:
        labels = targets.argmax(dim=-1).reshape(-1)
    else:
        labels = targets.reshape(-1)
    if labels.is_floating_point():
        rounded = labels.round()
        if not torch.allclose(labels, rounded):
            return {}
        labels = rounded
    labels = labels.detach().cpu().long().reshape(-1)
    unique, counts = torch.unique(labels, return_counts=True)
    return {int(k.item()): int(v.item()) for k, v in zip(unique, counts)}


class BatchInspector:
    """
    Inspect tensors and labels for common data problems.

    Usage:
        inspector = BatchInspector(expected_num_classes=10)
        report = inspector.inspect(inputs=x, targets=y, step=step)
        if not report.ok:
            print(report)
    """

    def __init__(
        self,
        *,
        expected_num_classes: Optional[int] = None,
        max_abs_value: float = 1.0e6,
        min_finite_fraction: float = 1.0,
        class_imbalance_warn: float = 0.95,
        constant_std_threshold: float = 1.0e-12,
    ) -> None:
        self.expected_num_classes = expected_num_classes
        self.max_abs_value = max_abs_value
        self.min_finite_fraction = min_finite_fraction
        self.class_imbalance_warn = class_imbalance_warn
        self.constant_std_threshold = constant_std_threshold

    def inspect(
        self,
        inputs: Any,
        targets: Optional[torch.Tensor] = None,
        *,
        step: int = 0,
    ) -> BatchReport:
        tensors = [_stats(name, tensor) for name, tensor in _iter_tensors(inputs, "inputs")]
        if targets is not None:
            tensors.extend(_stats(name, tensor) for name, tensor in _iter_tensors(targets, "targets"))

        warnings: List[str] = []
        for stat in tensors:
            if stat.numel == 0:
                warnings.append(f"{stat.name} is empty")
            if stat.finite_fraction < self.min_finite_fraction:
                warnings.append(
                    f"{stat.name} has non-finite values "
                    f"(nan={stat.nan_count}, inf={stat.inf_count})"
                )
            if stat.abs_max is not None and math.isfinite(stat.abs_max) and stat.abs_max > self.max_abs_value:
                warnings.append(f"{stat.name} abs max {stat.abs_max:.3g} exceeds {self.max_abs_value:.3g}")
            if stat.std is not None and stat.numel > 1 and stat.std <= self.constant_std_threshold:
                warnings.append(f"{stat.name} is nearly constant (std={stat.std:.3g})")

        distribution = _label_distribution(targets)
        imbalance_ratio: Optional[float] = None
        if distribution:
            total = sum(distribution.values())
            imbalance_ratio = max(distribution.values()) / max(1, total)
            if imbalance_ratio >= self.class_imbalance_warn and total > 1:
                warnings.append(f"dominant class covers {imbalance_ratio:.1%} of the batch")
            if self.expected_num_classes is not None:
                bad = [k for k in distribution if k < 0 or k >= self.expected_num_classes]
                if bad:
                    warnings.append(
                        f"labels outside [0, {self.expected_num_classes - 1}]: {bad[:8]}"
                    )

        return BatchReport(
            step=step,
            batch_size=_first_dim(inputs),
            tensors=tensors,
            label_distribution=distribution,
            imbalance_ratio=imbalance_ratio,
            warnings=warnings,
        )
