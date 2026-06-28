"""
Example training-dynamics diagnostics.

This module combines two paper-backed ideas that are useful in ordinary
classification runs but rarely exposed as a small library API:

* Example forgetting: Toneva et al., "An Empirical Study of Example Forgetting
  during Deep Neural Network Learning", ICLR 2019.
* Dataset cartography: Swayamdipta et al., "Dataset Cartography: Mapping and
  Diagnosing Datasets with Training Dynamics", EMNLP 2020.

Call update(...) during training with stable example ids, logits, and labels.
The tracker never touches model state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Optional, Sequence

import torch


ExampleId = Hashable


@dataclass
class ExampleDynamics:
    example_id: ExampleId
    seen: int = 0
    correct_count: int = 0
    forgetting_events: int = 0
    learning_events: int = 0
    first_learned_step: Optional[int] = None
    last_seen_step: Optional[int] = None
    last_correct: Optional[bool] = None
    confidence_sum: float = 0.0
    confidence_sq_sum: float = 0.0
    last_confidence: Optional[float] = None

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.seen if self.seen else 0.0

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.seen if self.seen else 0.0

    @property
    def confidence_variability(self) -> float:
        if self.seen < 2:
            return 0.0
        mean = self.mean_confidence
        var = max(0.0, self.confidence_sq_sum / self.seen - mean * mean)
        return math.sqrt(var)

    @property
    def unforgettable(self) -> bool:
        return self.first_learned_step is not None and self.forgetting_events == 0


@dataclass
class DynamicsSummary:
    step: int
    n_examples: int
    mean_accuracy: float
    mean_confidence: float
    mean_variability: float
    unforgettable_fraction: float
    never_learned_fraction: float
    total_forgetting_events: int
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"[step {self.step}] ExampleDynamics: examples={self.n_examples} "
            f"acc={self.mean_accuracy:.3f} conf={self.mean_confidence:.3f} "
            f"var={self.mean_variability:.3f}",
            f"  unforgettable={self.unforgettable_fraction:.1%} "
            f"never_learned={self.never_learned_fraction:.1%} "
            f"forget_events={self.total_forgetting_events}",
        ]
        for warning in self.warnings[:6]:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


class ExampleDynamicsTracker:
    """
    Track per-example learning dynamics from logits and stable example ids.

    Usage:
        tracker = ExampleDynamicsTracker()
        for step, (ids, x, y) in enumerate(loader):
            logits = model(x)
            tracker.update(ids, logits, y, step=step)

        print(tracker.summary(step))
        likely_label_noise = tracker.most_forgotten(50)
        ambiguous = tracker.cartography_region("ambiguous")
    """

    def __init__(
        self,
        *,
        easy_confidence: float = 0.75,
        hard_confidence: float = 0.35,
        ambiguous_variability: float = 0.15,
        min_observations: int = 3,
    ) -> None:
        self.easy_confidence = easy_confidence
        self.hard_confidence = hard_confidence
        self.ambiguous_variability = ambiguous_variability
        self.min_observations = min_observations
        self.examples: Dict[ExampleId, ExampleDynamics] = {}
        self._last_step: int = 0

    def update(
        self,
        example_ids: Sequence[ExampleId] | torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        step: int = 0,
    ) -> DynamicsSummary:
        if logits.dim() < 2:
            raise ValueError("logits must have shape (batch, classes)")
        batch = logits.shape[0]
        ids = self._normalise_ids(example_ids)
        if len(ids) != batch:
            raise ValueError(f"example_ids length {len(ids)} does not match batch size {batch}")
        if targets.shape[0] != batch:
            raise ValueError(f"targets batch size {targets.shape[0]} does not match logits batch size {batch}")

        with torch.no_grad():
            probs = torch.softmax(logits.detach().float(), dim=-1)
            labels = targets.detach().long().reshape(-1)
            preds = probs.argmax(dim=-1)
            true_conf = probs[torch.arange(batch, device=probs.device), labels].detach().cpu()
            correct = (preds == labels).detach().cpu()

        for idx, ex_id in enumerate(ids):
            state = self.examples.setdefault(ex_id, ExampleDynamics(example_id=ex_id))
            is_correct = bool(correct[idx].item())
            confidence = float(true_conf[idx].item())

            if state.last_correct is not None:
                if state.last_correct and not is_correct:
                    state.forgetting_events += 1
                elif not state.last_correct and is_correct:
                    state.learning_events += 1
            elif is_correct:
                state.learning_events += 1

            if is_correct and state.first_learned_step is None:
                state.first_learned_step = step

            state.seen += 1
            state.correct_count += int(is_correct)
            state.confidence_sum += confidence
            state.confidence_sq_sum += confidence * confidence
            state.last_confidence = confidence
            state.last_correct = is_correct
            state.last_seen_step = step

        self._last_step = step
        return self.summary(step=step)

    def summary(self, step: Optional[int] = None) -> DynamicsSummary:
        values = list(self.examples.values())
        n = len(values)
        if n == 0:
            return DynamicsSummary(
                step=self._last_step if step is None else step,
                n_examples=0, mean_accuracy=0.0, mean_confidence=0.0,
                mean_variability=0.0, unforgettable_fraction=0.0,
                never_learned_fraction=0.0, total_forgetting_events=0,
                warnings=["no examples recorded"],
            )

        learned = [ex for ex in values if ex.first_learned_step is not None]
        mean_accuracy = sum(ex.accuracy for ex in values) / n
        mean_confidence = sum(ex.mean_confidence for ex in values) / n
        mean_variability = sum(ex.confidence_variability for ex in values) / n
        unforgettable = sum(1 for ex in learned if ex.forgetting_events == 0)
        never_learned = n - len(learned)
        total_forgetting = sum(ex.forgetting_events for ex in values)
        warnings: List[str] = []
        if never_learned / n > 0.2:
            warnings.append(f"{never_learned / n:.1%} of tracked examples were never learned")
        if total_forgetting > n:
            warnings.append("forgetting events exceed number of examples; inspect top forgotten samples")

        return DynamicsSummary(
            step=self._last_step if step is None else step,
            n_examples=n,
            mean_accuracy=mean_accuracy,
            mean_confidence=mean_confidence,
            mean_variability=mean_variability,
            unforgettable_fraction=unforgettable / n,
            never_learned_fraction=never_learned / n,
            total_forgetting_events=total_forgetting,
            warnings=warnings,
        )

    def most_forgotten(self, k: int = 20) -> List[ExampleDynamics]:
        """Return examples with the most correct-to-incorrect transitions."""
        return sorted(
            self.examples.values(),
            key=lambda ex: (ex.forgetting_events, ex.confidence_variability, -ex.mean_confidence),
            reverse=True,
        )[:k]

    def never_learned(self) -> List[ExampleDynamics]:
        return [ex for ex in self.examples.values() if ex.first_learned_step is None]

    def unforgettable(self) -> List[ExampleDynamics]:
        return [ex for ex in self.examples.values() if ex.unforgettable]

    def cartography_region(self, region: str) -> List[ExampleDynamics]:
        """
        Return examples in a dataset-cartography-style region.

        Regions:
          easy: high confidence, low variability, mostly correct
          ambiguous: high confidence variability
          hard: low confidence or rarely correct
        """
        region = region.lower()
        if region not in {"easy", "ambiguous", "hard"}:
            raise ValueError("region must be one of: easy, ambiguous, hard")

        ready = [ex for ex in self.examples.values() if ex.seen >= self.min_observations]
        if region == "easy":
            return [
                ex for ex in ready
                if ex.mean_confidence >= self.easy_confidence
                and ex.confidence_variability < self.ambiguous_variability
                and ex.accuracy >= 0.8
            ]
        if region == "ambiguous":
            return [ex for ex in ready if ex.confidence_variability >= self.ambiguous_variability]
        return [
            ex for ex in ready
            if ex.mean_confidence <= self.hard_confidence or ex.accuracy <= 0.5
        ]

    @staticmethod
    def _normalise_ids(example_ids: Sequence[ExampleId] | torch.Tensor) -> List[ExampleId]:
        if isinstance(example_ids, torch.Tensor):
            flat = example_ids.detach().cpu().reshape(-1)
            return [int(x.item()) for x in flat]
        return list(example_ids)
