"""
Early example-importance diagnostics.

Paul, Ganguli, and Dziugaite (NeurIPS 2021) proposed EL2N and GraNd scores for
finding important examples early in training. This module implements the cheap
EL2N side: ||softmax(logits) - one_hot(label)||_2, tracked per example.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence

import torch
import torch.nn.functional as F


ExampleId = Hashable


@dataclass
class EL2NExample:
    example_id: ExampleId
    count: int = 0
    score_sum: float = 0.0
    score_sq_sum: float = 0.0
    last_score: Optional[float] = None

    @property
    def score(self) -> float:
        return self.score_sum / self.count if self.count else 0.0

    @property
    def score_std(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.score
        return math.sqrt(max(0.0, self.score_sq_sum / self.count - mean * mean))


@dataclass
class EL2NSummary:
    step: int
    n_examples: int
    mean_score: float
    high_score_fraction: float

    def __str__(self) -> str:
        return (
            f"[step {self.step}] EL2NTracker: examples={self.n_examples} "
            f"mean={self.mean_score:.3f} high={self.high_score_fraction:.1%}"
        )


def el2n_scores(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return per-example EL2N scores for a batch."""
    if logits.dim() != 2:
        raise ValueError("logits must have shape (batch, classes)")
    labels = targets.detach().long().reshape(-1)
    if labels.numel() != logits.shape[0]:
        raise ValueError("targets must contain one label per logit row")
    probs = torch.softmax(logits.detach().float(), dim=-1)
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).to(probs.device).float()
    return (probs - one_hot).norm(dim=1).detach().cpu()


class EL2NTracker:
    """
    Track early-learning EL2N scores per example.

    High EL2N examples are often important, hard, mislabeled, or distribution
    edge cases. Low EL2N examples are candidates for pruning experiments.
    """

    def __init__(self, *, high_score_threshold: float = 1.0, min_observations: int = 1) -> None:
        self.high_score_threshold = high_score_threshold
        self.min_observations = min_observations
        self.examples: Dict[ExampleId, EL2NExample] = {}
        self._last_step = 0

    def update(
        self,
        example_ids: Sequence[ExampleId] | torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        step: int = 0,
    ) -> EL2NSummary:
        ids = self._normalise_ids(example_ids)
        scores = el2n_scores(logits, targets)
        if len(ids) != scores.numel():
            raise ValueError(f"example_ids length {len(ids)} does not match batch size {scores.numel()}")
        for ex_id, score_tensor in zip(ids, scores):
            score = float(score_tensor.item())
            state = self.examples.setdefault(ex_id, EL2NExample(example_id=ex_id))
            state.count += 1
            state.score_sum += score
            state.score_sq_sum += score * score
            state.last_score = score
        self._last_step = step
        return self.summary(step=step)

    def summary(self, step: Optional[int] = None) -> EL2NSummary:
        ready = [ex for ex in self.examples.values() if ex.count >= self.min_observations]
        if not ready:
            return EL2NSummary(step=self._last_step if step is None else step,
                               n_examples=len(self.examples), mean_score=0.0,
                               high_score_fraction=0.0)
        high = [ex for ex in ready if ex.score >= self.high_score_threshold]
        return EL2NSummary(step=self._last_step if step is None else step,
                           n_examples=len(self.examples),
                           mean_score=sum(ex.score for ex in ready) / len(ready),
                           high_score_fraction=len(high) / len(ready))

    def highest(self, k: int = 20) -> List[EL2NExample]:
        ready = [ex for ex in self.examples.values() if ex.count >= self.min_observations]
        return sorted(ready, key=lambda ex: (ex.score, ex.score_std), reverse=True)[:k]

    def lowest(self, k: int = 20) -> List[EL2NExample]:
        ready = [ex for ex in self.examples.values() if ex.count >= self.min_observations]
        return sorted(ready, key=lambda ex: (ex.score, -ex.score_std))[:k]

    @staticmethod
    def _normalise_ids(example_ids: Sequence[ExampleId] | torch.Tensor) -> List[ExampleId]:
        if isinstance(example_ids, torch.Tensor):
            return [int(x.item()) for x in example_ids.detach().cpu().reshape(-1)]
        return list(example_ids)
