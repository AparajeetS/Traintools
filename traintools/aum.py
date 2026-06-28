"""
Area Under the Margin (AUM) diagnostics.

Pleiss et al. 2020 showed that the average true-class margin over training is a
strong signal for mislabeled or ambiguous examples. This module implements the
core statistic as a lightweight tracker. The full paper's threshold-sample
procedure is optional research protocol; this utility exposes the ranking signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence

import torch


ExampleId = Hashable


@dataclass
class AUMExample:
    example_id: ExampleId
    count: int = 0
    margin_sum: float = 0.0
    margin_sq_sum: float = 0.0
    min_margin: float = float("inf")
    max_margin: float = float("-inf")
    last_margin: Optional[float] = None

    @property
    def aum(self) -> float:
        return self.margin_sum / self.count if self.count else 0.0

    @property
    def margin_std(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.aum
        return math.sqrt(max(0.0, self.margin_sq_sum / self.count - mean * mean))


@dataclass
class AUMSummary:
    step: int
    n_examples: int
    mean_aum: float
    low_aum_fraction: float
    warnings: List[str]

    def __str__(self) -> str:
        status = "OK" if not self.warnings else "WARN"
        lines = [
            f"[step {self.step}] AUMTracker: {status} examples={self.n_examples} "
            f"mean_aum={self.mean_aum:.3f} low_aum={self.low_aum_fraction:.1%}"
        ]
        for warning in self.warnings[:6]:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


class AUMTracker:
    """
    Track Area Under the Margin for each training example.

    The per-step margin is:
        true_class_logit - max(other_class_logits)

    Low AUM examples are candidates for label audit, ambiguity review, or
    down-weighting experiments.
    """

    def __init__(self, *, low_aum_threshold: float = 0.0, min_observations: int = 3) -> None:
        self.low_aum_threshold = low_aum_threshold
        self.min_observations = min_observations
        self.examples: Dict[ExampleId, AUMExample] = {}
        self._last_step = 0

    def update(
        self,
        example_ids: Sequence[ExampleId] | torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        step: int = 0,
    ) -> AUMSummary:
        if logits.dim() != 2:
            raise ValueError("logits must have shape (batch, classes)")
        ids = self._normalise_ids(example_ids)
        batch, n_classes = logits.shape
        if n_classes < 2:
            raise ValueError("AUM requires at least two classes")
        if len(ids) != batch:
            raise ValueError(f"example_ids length {len(ids)} does not match batch size {batch}")
        labels = targets.detach().long().reshape(-1)
        if labels.numel() != batch:
            raise ValueError(f"targets size {labels.numel()} does not match batch size {batch}")

        with torch.no_grad():
            scores = logits.detach().float().clone()
            true_logits = scores[torch.arange(batch, device=scores.device), labels]
            scores[torch.arange(batch, device=scores.device), labels] = float("-inf")
            other_logits = scores.max(dim=1).values
            margins = (true_logits - other_logits).detach().cpu()

        for ex_id, margin_tensor in zip(ids, margins):
            margin = float(margin_tensor.item())
            state = self.examples.setdefault(ex_id, AUMExample(example_id=ex_id))
            state.count += 1
            state.margin_sum += margin
            state.margin_sq_sum += margin * margin
            state.min_margin = min(state.min_margin, margin)
            state.max_margin = max(state.max_margin, margin)
            state.last_margin = margin

        self._last_step = step
        return self.summary(step=step)

    def summary(self, step: Optional[int] = None) -> AUMSummary:
        ready = [ex for ex in self.examples.values() if ex.count >= self.min_observations]
        if not self.examples:
            return AUMSummary(step=self._last_step if step is None else step,
                              n_examples=0, mean_aum=0.0, low_aum_fraction=0.0,
                              warnings=["no examples recorded"])
        denom = max(1, len(ready))
        mean_aum = sum(ex.aum for ex in ready) / denom if ready else 0.0
        low = [ex for ex in ready if ex.aum <= self.low_aum_threshold]
        warnings: List[str] = []
        if ready and len(low) / len(ready) > 0.1:
            warnings.append(f"{len(low) / len(ready):.1%} of ready examples have low AUM")
        return AUMSummary(step=self._last_step if step is None else step,
                          n_examples=len(self.examples), mean_aum=mean_aum,
                          low_aum_fraction=len(low) / denom, warnings=warnings)

    def lowest_aum(self, k: int = 20) -> List[AUMExample]:
        ready = [ex for ex in self.examples.values() if ex.count >= self.min_observations]
        return sorted(ready, key=lambda ex: (ex.aum, -ex.margin_std))[:k]

    def suspicious(self, *, threshold: Optional[float] = None) -> List[AUMExample]:
        cutoff = self.low_aum_threshold if threshold is None else threshold
        return [ex for ex in self.examples.values()
                if ex.count >= self.min_observations and ex.aum <= cutoff]

    @staticmethod
    def _normalise_ids(example_ids: Sequence[ExampleId] | torch.Tensor) -> List[ExampleId]:
        if isinstance(example_ids, torch.Tensor):
            return [int(x.item()) for x in example_ids.detach().cpu().reshape(-1)]
        return list(example_ids)
