"""
Raw PyTorch training-loop integration for traintools.

Two GNS modes:

  (A) Extra-pass mode (default) — call tracker.step(...) with the current batch;
      GNS is estimated with a few extra eval-mode forward/backward passes.

        tracker = TraintoolsTracker(model, loss_fn)
        for step, (x, y) in enumerate(loader):
            loss = loss_fn(model(x), y); loss.backward()
            optimizer.step(); optimizer.zero_grad()
            tracker.step(step=step, inputs=x, targets=y, val_loss=val_loss)

  (B) Free accumulation mode — if you already use gradient accumulation, GNS is
      computed at ZERO extra cost from the micro-batch gradients you compute anyway:

        tracker = TraintoolsTracker(model, loss_fn, gns_free_accum=True,
                                    micro_batch_size=B_micro)
        for step in range(num_steps):
            for micro in micro_batches:
                (loss_fn(model(xm), ym) / n_accum).backward()
                tracker.record_microbatch()          # after each backward
            optimizer.step()
            tracker.step(step=step, val_loss=val_loss)  # no inputs/targets needed
            optimizer.zero_grad()
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from traintools.earlyguard import EarlyStopDecision, TrainGuard
from traintools.gradnoise import (
    GNSEstimator, GNSHistory, GradientAccumulationGNS, estimate_gns,
)
from traintools.plasticity import PlasticityHistory, PlasticityProbe


class TraintoolsTracker:
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
        gns_splits: int = 2,
        gns_ema_decay: float = 0.95,
        gns_free_accum: bool = False,
        micro_batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.gns_freq = gns_freq
        self.plasticity_freq = plasticity_freq
        self.per_layer_gns = per_layer_gns
        self.gns_splits = gns_splits
        self.verbose = verbose

        self._probe = PlasticityProbe(model)
        self._gns_history = GNSHistory()
        self._plasticity_history = PlasticityHistory()
        self._guard = TrainGuard(
            min_improvement=min_improvement,
            patience_steps=patience_steps,
            horizon_steps=horizon_steps,
        ) if earlyguard else None

        # Free accumulation path
        self.gns_free_accum = gns_free_accum
        if gns_free_accum:
            if micro_batch_size is None:
                raise ValueError("micro_batch_size is required when gns_free_accum=True")
            self._accum_gns = GradientAccumulationGNS(model, micro_batch_size, decay=gns_ema_decay)
        else:
            self._accum_gns = None
            # EMA estimator for the extra-pass path lives inside estimate_gns calls;
            # we keep a persistent one to smooth across calls.
            self._gns_estimator = GNSEstimator(decay=gns_ema_decay)

    # ── Free-accumulation hook ────────────────────────────────────────────────

    def record_microbatch(self) -> None:
        """Free-accumulation mode: call after each micro-batch backward()."""
        if self._accum_gns is not None:
            self._accum_gns.record_microbatch()

    # ── Main per-step entry point ─────────────────────────────────────────────

    def step(
        self,
        step: int,
        inputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        val_loss: Optional[float] = None,
    ) -> Optional[EarlyStopDecision]:
        # ── GNS ──────────────────────────────────────────────────────────────
        if self._accum_gns is not None:
            # Free path: compute from recorded micro-batches every gns_freq steps.
            if step > 0 and step % self.gns_freq == 0:
                result = self._accum_gns.compute(step=step)
                if result is not None:
                    self._gns_history.record(result)
                    if self.verbose:
                        print(f"\n[traintools:GNS]\n{result}")
            self._accum_gns.reset_accumulation()
        elif (inputs is not None and targets is not None
              and step > 0 and step % self.gns_freq == 0):
            try:
                # Pause the plasticity probe so GNS's diagnostic forward passes
                # don't pollute the activation buffer.
                with self._probe.paused():
                    raw = estimate_gns(
                        self.model, self.loss_fn, inputs, targets,
                        n_splits=self.gns_splits, step=step, per_layer=self.per_layer_gns,
                    )
                # Smooth across calls via the persistent EMA estimator.
                smoothed = self._gns_estimator.update_from_estimates(
                    raw.tr_sigma, raw.g_squared, raw.current_batch,
                    step=step, per_layer=raw.per_layer,
                )
                self._gns_history.record(smoothed)
                if self.verbose:
                    print(f"\n[traintools:GNS]\n{smoothed}")
            except Exception as e:
                if self.verbose:
                    print(f"[traintools:GNS] estimation failed: {e}")

        # ── Plasticity ─────────────────────────────────────────────────────────
        if step > 0 and step % self.plasticity_freq == 0:
            result_p = self._probe.measure(step=step)
            self._plasticity_history.record(result_p)
            self._probe.reset_buffers()
            if self.verbose:
                print(f"\n[traintools:Plasticity]\n{result_p}")

        # ── TrainGuard ───────────────────────────────────────────────────────
        if self._guard is not None and val_loss is not None:
            self._guard.record(step=step, val_loss=val_loss)
            decision = self._guard.evaluate()
            if self.verbose and step % self.gns_freq == 0:
                print(f"\n[traintools:TrainGuard]\n{decision}")
            if decision.should_stop:
                return decision
        return None

    # ── Accessors / plotting ──────────────────────────────────────────────────

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
