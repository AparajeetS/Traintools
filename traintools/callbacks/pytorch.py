"""
Raw PyTorch training loop integration for traintools.

For users not using HuggingFace Trainer.

Usage:
    from traintools.callbacks.pytorch import TraintoolsTracker

    tracker = TraintoolsTracker(
        model=model,
        loss_fn=criterion,
        gns_freq=200,
        plasticity_freq=200,
        earlyguard=True,
    )

    for epoch in range(epochs):
        for step, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tracker.step(
                step=global_step,
                inputs=inputs,
                targets=targets,
                val_loss=val_loss,  # optional, pass when available
            )
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from traintools.earlyguard import EarlyStopDecision, TrainGuard
from traintools.gradnoise import GNSHistory, GNSResult, estimate_gns
from traintools.plasticity import PlasticityHistory, PlasticityProbe, PlasticityResult


class TraintoolsTracker:
    """
    Convenience wrapper for raw PyTorch training loops.
    Tracks GNS, plasticity, and optional early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        gns_freq: int = 200,
        plasticity_freq: int = 200,
        earlyguard: bool = True,
        min_improvement: float = 1e-4,
        patience_steps: int = 1000,
        horizon_steps: int = 500,
        per_layer_gns: bool = False,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.gns_freq = gns_freq
        self.plasticity_freq = plasticity_freq
        self.per_layer_gns = per_layer_gns
        self.verbose = verbose

        self._probe = PlasticityProbe(model)
        self._gns_history = GNSHistory()
        self._plasticity_history = PlasticityHistory()
        self._guard = TrainGuard(
            min_improvement=min_improvement,
            patience_steps=patience_steps,
            horizon_steps=horizon_steps,
        ) if earlyguard else None

        self._plasticity_window_start = 0

    def step(
        self,
        step: int,
        inputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        val_loss: Optional[float] = None,
    ) -> Optional[EarlyStopDecision]:
        """
        Call once per optimizer step.

        Args:
            step:     current global training step
            inputs:   current batch inputs (needed for GNS estimation)
            targets:  current batch targets (needed for GNS estimation)
            val_loss: current validation loss (needed for TrainGuard)

        Returns:
            EarlyStopDecision if TrainGuard fires, else None.
        """
        # ── GNS ───────────────────────────────────────────────────────────────
        if (inputs is not None and targets is not None and
                step > 0 and step % self.gns_freq == 0):
            try:
                result = estimate_gns(
                    model=self.model,
                    loss_fn=self.loss_fn,
                    inputs=inputs,
                    targets=targets,
                    step=step,
                    per_layer=self.per_layer_gns,
                )
                self._gns_history.record(result)
                if self.verbose:
                    print(f"\n[traintools:GNS]\n{result}")
            except Exception as e:
                if self.verbose:
                    print(f"[traintools:GNS] estimation failed: {e}")

        # ── Plasticity ────────────────────────────────────────────────────────
        if step > 0 and step % self.plasticity_freq == 0:
            result_p = self._probe.measure(step=step)
            self._plasticity_history.record(result_p)
            self._probe.reset_activation_buffer()
            if self.verbose:
                print(f"\n[traintools:Plasticity]\n{result_p}")

        # ── TrainGuard ────────────────────────────────────────────────────────
        if self._guard is not None and val_loss is not None:
            self._guard.record(step=step, val_loss=val_loss)
            decision = self._guard.evaluate()
            if self.verbose and step % self.gns_freq == 0:
                print(f"\n[traintools:TrainGuard]\n{decision}")
            if decision.should_stop:
                return decision

        return None

    @property
    def gns_history(self) -> GNSHistory:
        return self._gns_history

    @property
    def plasticity_history(self) -> PlasticityHistory:
        return self._plasticity_history

    @property
    def guard(self) -> Optional[TrainGuard]:
        return self._guard

    def plot(self):
        """Plot all three metrics side by side. Requires matplotlib."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 3))
        self._gns_history.plot(ax=axes[0])
        self._plasticity_history.plot(ax=axes[1])
        if self._guard and self._guard._steps:
            self._guard.plot(ax=axes[2])
        else:
            axes[2].set_visible(False)
        plt.tight_layout()
        return fig
