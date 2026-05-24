"""
Plasticity Probe — detects loss of plasticity during training.

Networks trained for long periods, or under continual / repeated fine-tuning,
gradually lose the ability to adapt. The phenomenon is now well documented:
the failure shows up in the *representations*, as

  1. Dormant (dead) units — neurons whose activation is ~0 for every input, so
     they carry no information and receive no gradient.
  2. Collapsed feature rank — the layer's activations span far fewer effective
     dimensions than it has units, i.e. the representation has degenerated.

This module measures both directly on the activations (not on weight matrices),
matching the operational definitions used in the loss-of-plasticity literature.

For each activation module we compute, on a sample of activations:
  dead_fraction  — fraction of units with max|activation| below a threshold
  feature_erank  — effective rank of the activation covariance, normalised to
                   [0,1] by the number of units. effective rank is
                   exp(H(σ/Σσ)) where σ are the singular values of the centred
                   activation matrix; H is Shannon entropy.

The per-module Plasticity Score is the geometric mean of (1 − dead_fraction)
and feature_erank; the global score is the geometric mean across modules.

References:
  Dohare, Sutton et al. 2024, "Loss of plasticity in deep continual learning",
    Nature 632, 768–774.
  Lyle et al. 2023, "Understanding plasticity in neural networks", ICML.
  Kumar et al. 2023, "Maintaining plasticity via regenerative regularization".

The normalised feature_erank is identical to the activation-covariance
effective-rank probe (erank(Σ_l)/n_l) studied as a single-SVD representation
metric; here it is repurposed as a plasticity signal.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class LayerPlasticityResult:
    name: str
    dead_fraction: float       # fraction of dormant units in this activation
    feature_erank: float       # effective rank of activations, normalised to [0,1]
    n_units: int
    n_samples: int
    score: float               # geometric mean of (1−dead) and feature_erank
    flags: List[str]

    def is_critical(self) -> bool:
        return self.score < 0.3

    def __str__(self) -> str:
        flag_str = " | ".join(self.flags) if self.flags else "healthy"
        return (f"{self.name}: score={self.score:.3f}  dead={self.dead_fraction:.1%}  "
                f"erank={self.feature_erank:.3f}  units={self.n_units}  [{flag_str}]")


@dataclass
class PlasticityResult:
    step: int
    global_score: float
    layers: List[LayerPlasticityResult] = field(default_factory=list)
    critical_layers: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"[step {self.step}] Plasticity Score: {self.global_score:.3f}"]
        if self.critical_layers:
            lines.append(f"  ! Critical layers: {', '.join(self.critical_layers)}")
            lines.append("  Action: reinitialise dormant units (continual backprop) "
                         "or add regenerative regularisation.")
        else:
            lines.append("  All layers healthy.")
        for lr in self.layers:
            if lr.is_critical():
                lines.append(f"  {lr}")
        return "\n".join(lines)


# ── Effective rank of an activation matrix ───────────────────────────────────

def activation_effective_rank(A: torch.Tensor) -> float:
    """
    Normalised effective rank of an (N_samples, N_units) activation matrix.

    Centres the activations, takes singular values, and returns
    exp(entropy(σ / Σσ)) / N_units  ∈ (0, 1].
    1.0 = activations isotropically span all units; →0 = collapse to few dims.
    """
    if A.dim() != 2:
        A = A.reshape(A.shape[0], -1)
    n_samples, n_units = A.shape
    if n_units <= 1 or n_samples < 2:
        return 1.0
    with torch.no_grad():
        A = A.float()
        A = A - A.mean(0, keepdim=True)
        try:
            s = torch.linalg.svdvals(A)
        except Exception:
            return 1.0
        s = s[s > 1e-10]
        if s.numel() == 0:
            return 0.0
        p = s / s.sum()
        entropy = -(p * torch.log(p + 1e-12)).sum().item()
        return math.exp(entropy) / n_units


def _layer_score(dead_fraction: float, feature_erank: float) -> tuple[float, List[str]]:
    flags: List[str] = []
    if dead_fraction > 0.5:
        flags.append(f"DORMANT>{dead_fraction:.0%}")
    elif dead_fraction > 0.2:
        flags.append(f"dormant>{dead_fraction:.0%}")
    if feature_erank < 0.1:
        flags.append("rank-collapsed")
    elif feature_erank < 0.3:
        flags.append("low-rank")
    alive = max(0.0, 1.0 - dead_fraction)
    score = math.sqrt(alive * feature_erank)   # geometric mean of two [0,1] signals
    return score, flags


# ── Probe ────────────────────────────────────────────────────────────────────

class PlasticityProbe:
    """
    Attach to a model to track plasticity during training.

    Registers forward hooks on activation modules and accumulates a capped
    sample of their outputs. At measure() time it computes dormant-unit fraction
    and feature effective rank per activation module — attributed directly to the
    module that produced them (no name-matching heuristics).

    Usage:
        probe = PlasticityProbe(model)
        # run some forward passes (training or a dedicated probe batch) ...
        result = probe.measure(step=step)
        probe.reset_buffers()   # start a fresh measurement window
    """

    def __init__(
        self,
        model: nn.Module,
        activation_types: tuple = (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.Tanh, nn.ELU),
        dead_threshold: float = 1e-3,
        max_samples: int = 1024,
    ) -> None:
        self.model = model
        self.dead_threshold = dead_threshold
        self.max_samples = max_samples
        self._buffers: Dict[str, torch.Tensor] = {}   # name -> (n_samples, n_units)
        self._hooks: list = []
        self._paused = False
        self._register(activation_types)

    def _register(self, activation_types: tuple) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, activation_types):
                self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            if self._paused:
                return
            with torch.no_grad():
                a = out.detach()
                # Reshape to (samples, units): treat channels/features as units,
                # everything else (batch, spatial) as samples.
                if a.dim() == 1:
                    rows = a.reshape(1, -1)
                elif a.dim() == 2:               # (B, U)
                    rows = a
                else:                            # (B, C, ...) conv: units = C
                    C = a.shape[1]
                    rows = a.movedim(1, -1).reshape(-1, C)
                rows = rows.float().cpu()
                if name in self._buffers:
                    rows = torch.cat([self._buffers[name], rows], dim=0)
                # Cap stored samples
                if rows.shape[0] > self.max_samples:
                    rows = rows[-self.max_samples:]
                self._buffers[name] = rows
        return hook

    def reset_buffers(self) -> None:
        """Clear accumulated activations. Call at the start of each window."""
        self._buffers.clear()

    @contextmanager
    def paused(self):
        """Temporarily stop capturing activations (e.g. during GNS diagnostic passes)."""
        prev = self._paused
        self._paused = True
        try:
            yield
        finally:
            self._paused = prev

    # Backwards-compatible alias
    def reset_activation_buffer(self) -> None:
        self.reset_buffers()

    def measure(self, step: int = 0) -> PlasticityResult:
        layers: List[LayerPlasticityResult] = []
        for name, A in self._buffers.items():
            if A.numel() == 0:
                continue
            n_samples, n_units = A.shape
            # Dormant fraction: units whose max magnitude across samples is ~0
            unit_max = A.abs().max(0).values
            dead_fraction = float((unit_max < self.dead_threshold).float().mean())
            # Feature effective rank
            erank = activation_effective_rank(A)
            score, flags = _layer_score(dead_fraction, erank)
            layers.append(LayerPlasticityResult(
                name=name, dead_fraction=dead_fraction, feature_erank=erank,
                n_units=n_units, n_samples=n_samples, score=score, flags=flags,
            ))

        if not layers:
            global_score = 1.0
        else:
            log_sum = sum(math.log(max(lr.score, 1e-12)) for lr in layers)
            global_score = math.exp(log_sum / len(layers))

        critical = [lr.name for lr in layers if lr.is_critical()]
        return PlasticityResult(step=step, global_score=global_score,
                                layers=layers, critical_layers=critical)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


# ── History ──────────────────────────────────────────────────────────────────

class PlasticityHistory:
    def __init__(self) -> None:
        self.results: List[PlasticityResult] = []

    def record(self, result: PlasticityResult) -> None:
        self.results.append(result)

    @property
    def steps(self) -> List[int]:
        return [r.step for r in self.results]

    @property
    def scores(self) -> List[float]:
        return [r.global_score for r in self.results]

    def is_degrading(self, window: int = 5, threshold: float = 0.05) -> bool:
        s = self.scores[-window:]
        if len(s) < 2:
            return False
        return (s[0] - s[-1]) / (s[0] + 1e-12) > threshold

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self.steps, self.scores, marker="o", linewidth=1.5,
                color="orange", label="Plasticity Score")
        ax.axhline(0.3, color="red", linestyle="--", linewidth=0.8, label="critical threshold")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Plasticity Score")
        ax.set_title("Network Plasticity (feature rank + dormant units)")
        ax.legend(fontsize=8)
        return ax
