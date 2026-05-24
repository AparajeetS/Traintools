"""
Plasticity Probe — detects loss of plasticity during training.

Networks trained for long periods (or via continual learning / fine-tuning)
gradually lose the ability to adapt. Symptoms:
  - Dead neurons: never activate, contribute nothing
  - Collapsed weight spectra: weights become nearly rank-1
  - Frozen layers: gradient magnitude falls far below weight magnitude

PlasticityProbe hooks into forward/backward passes and computes a
per-layer Plasticity Score ∈ [0, 1], where 1 = fully plastic, 0 = dead.

Global score is the geometric mean across layers.

References:
  Lyle et al. 2023 "Understanding Plasticity in Neural Networks" (DeepMind)
  Kumar et al. 2023 "Maintaining Plasticity in Continual Learning" (Google)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import torch
import torch.nn as nn


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LayerPlasticityResult:
    name: str
    dead_fraction: float       # fraction of neurons dead (activation ≡ 0 on all batch samples)
    erank: float               # effective rank of weight matrix, normalized to [0,1]
    grad_weight_ratio: float   # ||grad||_F / ||weight||_F
    score: float               # geometric mean → [0, 1]
    flags: List[str]           # human-readable warnings

    def is_critical(self) -> bool:
        return self.score < 0.3

    def __str__(self) -> str:
        flag_str = " | ".join(self.flags) if self.flags else "healthy"
        return (f"{self.name}: score={self.score:.3f}  "
                f"dead={self.dead_fraction:.1%}  erank={self.erank:.3f}  "
                f"gw_ratio={self.grad_weight_ratio:.2e}  [{flag_str}]")


@dataclass
class PlasticityResult:
    step: int
    global_score: float
    layers: List[LayerPlasticityResult] = field(default_factory=list)
    critical_layers: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"[step {self.step}] Plasticity Score: {self.global_score:.3f}",
        ]
        if self.critical_layers:
            lines.append(f"  ⚠ Critical layers: {', '.join(self.critical_layers)}")
            lines.append("  Action: consider layer re-initialization or reduced LR warm-up.")
        else:
            lines.append("  All layers healthy.")
        for lr in self.layers:
            if lr.is_critical():
                lines.append(f"  {lr}")
        return "\n".join(lines)


# ── Core computation ───────────────────────────────────────────────────────────

def _effective_rank_normalized(W: torch.Tensor) -> float:
    """
    Effective rank of W, normalized to [0, 1].
    erank = exp(H(σ/||σ||_1)) / min(m, n)
    where σ are singular values of W.
    """
    if W.dim() > 2:
        W = W.flatten(1)  # flatten conv kernels
    m, n = W.shape
    rank_max = min(m, n)
    if rank_max == 1:
        return 1.0

    with torch.no_grad():
        try:
            # Use fast randomized SVD if available
            s = torch.linalg.svdvals(W.float())
        except Exception:
            return 1.0

        s = s[s > 1e-10]
        if len(s) == 0:
            return 0.0

        s_norm = s / s.sum()
        # Shannon entropy → effective rank
        entropy = -(s_norm * torch.log(s_norm + 1e-12)).sum().item()
        erank = math.exp(entropy)
        return erank / rank_max


def _compute_layer_score(
    dead_fraction: float,
    erank_norm: float,
    gw_ratio: float,
    gw_reference: float = 1e-3,  # healthy ratio baseline
) -> tuple[float, list[str]]:
    """Geometric mean of three sub-scores, with flags."""
    flags = []

    # Sub-score 1: dead neuron fraction (1 = no dead neurons)
    dead_score = max(0.0, 1.0 - dead_fraction)
    if dead_fraction > 0.5:
        flags.append(f"DEAD>{dead_fraction:.0%}")
    elif dead_fraction > 0.2:
        flags.append(f"dead>{dead_fraction:.0%}")

    # Sub-score 2: effective rank (already in [0,1])
    rank_score = erank_norm
    if erank_norm < 0.1:
        flags.append("rank-collapsed")

    # Sub-score 3: gradient-to-weight magnitude ratio
    # Healthy: gw_ratio ≈ 1e-3 to 1e-1
    # Dead:    gw_ratio < 1e-6
    if gw_ratio < 1e-9:
        gw_score = 0.0
        flags.append("gradient-frozen")
    elif gw_ratio < gw_reference * 0.01:
        gw_score = 0.1
        flags.append("low-gradient")
    else:
        # Sigmoid-style mapping: rises from 0 to 1 as gw_ratio → gw_reference
        gw_score = min(1.0, gw_ratio / gw_reference)

    # Geometric mean (any 0 → score is 0)
    score = (dead_score * rank_score * gw_score) ** (1.0 / 3.0)
    return score, flags


# ── Probe class ────────────────────────────────────────────────────────────────

class PlasticityProbe:
    """
    Attach to a model to track plasticity during training.

    Usage:
        probe = PlasticityProbe(model)
        # ... training loop ...
        if step % 200 == 0:
            result = probe.measure(step=step)
            print(result)

    The probe registers forward hooks on activation layers to count dead neurons,
    and reads weight/gradient tensors at measurement time.
    """

    def __init__(
        self,
        model: nn.Module,
        activation_types: tuple = (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU),
        dead_threshold: float = 1e-6,
    ) -> None:
        self.model = model
        self.dead_threshold = dead_threshold
        self._activation_buffer: Dict[str, torch.Tensor] = {}
        self._hooks: list = []
        self._register_hooks(activation_types)

    def _register_hooks(self, activation_types: tuple) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, activation_types):
                # Capture the OUTPUT of each activation layer
                def make_hook(n: str):
                    def hook(mod, inp, out):
                        # Accumulate max activation per neuron over batch dim
                        with torch.no_grad():
                            # out shape: (B, ...) — max over batch, keep spatial
                            abs_out = out.detach().abs()
                            # Flatten everything except the channel/neuron dim
                            if abs_out.dim() > 2:
                                # (B, C, ...) → max over (B, ...), keep C
                                channel_max = abs_out.flatten(2).max(-1).values.max(0).values
                            else:
                                channel_max = abs_out.max(0).values
                            if n in self._activation_buffer:
                                self._activation_buffer[n] = torch.maximum(
                                    self._activation_buffer[n], channel_max
                                )
                            else:
                                self._activation_buffer[n] = channel_max
                    return hook
                self._hooks.append(module.register_forward_hook(make_hook(name)))

    def reset_activation_buffer(self) -> None:
        """Call at the start of each measurement window to reset dead-neuron counters."""
        self._activation_buffer.clear()

    def measure(self, step: int = 0) -> PlasticityResult:
        """
        Compute plasticity scores using current weights and accumulated activations.
        Call after at least one forward pass since the last reset_activation_buffer().
        """
        layer_results: list[LayerPlasticityResult] = []

        # Build a lookup from activation buffer names to their parent linear layers
        # Also collect standalone linear/conv layers without captured activations
        measured_names: Set[str] = set()

        for name, module in self.model.named_modules():
            W = getattr(module, "weight", None)
            if W is None or W.dim() < 2:
                continue

            # Dead fraction: use activation buffer if this layer feeds into one
            dead_frac = self._dead_fraction_for_layer(name)

            # Effective rank
            erank = _effective_rank_normalized(W)

            # Gradient-to-weight ratio
            if module.weight.grad is not None:
                gw = (module.weight.grad.norm().item() /
                      (module.weight.norm().item() + 1e-12))
            else:
                gw = 0.0

            score, flags = _compute_layer_score(dead_frac, erank, gw)
            layer_results.append(LayerPlasticityResult(
                name=name,
                dead_fraction=dead_frac,
                erank=erank,
                grad_weight_ratio=gw,
                score=score,
                flags=flags,
            ))
            measured_names.add(name)

        if not layer_results:
            global_score = 1.0
        else:
            scores = [lr.score for lr in layer_results]
            # Geometric mean
            log_sum = sum(math.log(s + 1e-12) for s in scores)
            global_score = math.exp(log_sum / len(scores))

        critical = [lr.name for lr in layer_results if lr.is_critical()]

        return PlasticityResult(
            step=step,
            global_score=global_score,
            layers=layer_results,
            critical_layers=critical,
        )

    def _dead_fraction_for_layer(self, layer_name: str) -> float:
        """
        Estimate dead neuron fraction.
        Heuristic: look for an activation buffer entry that starts with this layer's name.
        Falls back to 0 (assume healthy) if no activation data captured.
        """
        for buf_name, max_acts in self._activation_buffer.items():
            if buf_name.startswith(layer_name) or layer_name in buf_name:
                dead = (max_acts < self.dead_threshold).float().mean().item()
                return dead
        return 0.0

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


class PlasticityHistory:
    """Running log of plasticity measurements."""

    def __init__(self) -> None:
        self.results: list[PlasticityResult] = []

    def record(self, result: PlasticityResult) -> None:
        self.results.append(result)

    @property
    def steps(self) -> list[int]:
        return [r.step for r in self.results]

    @property
    def scores(self) -> list[float]:
        return [r.global_score for r in self.results]

    def is_degrading(self, window: int = 5, threshold: float = 0.05) -> bool:
        """True if score has dropped by more than threshold over the last window measurements."""
        s = self.scores[-window:]
        if len(s) < 2:
            return False
        return (s[0] - s[-1]) / (s[0] + 1e-12) > threshold

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self.steps, self.scores, marker="o", linewidth=1.5, color="orange", label="Plasticity Score")
        ax.axhline(0.3, color="red", linestyle="--", linewidth=0.8, label="critical threshold")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Plasticity Score")
        ax.set_title("Network Plasticity")
        ax.legend(fontsize=8)
        return ax
