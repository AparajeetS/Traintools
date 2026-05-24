"""
HuggingFace Trainer callback for traintools.

Computes Gradient Noise Scale, Plasticity Score, and TrainGuard early stopping
inside a standard `transformers.Trainer` run — no Trainer subclass required.

    from transformers import Trainer
    from traintools.callbacks.huggingface import TraintoolsCallback

    trainer = Trainer(
        model=model, ...,
        callbacks=[TraintoolsCallback(gns_freq=200, plasticity_freq=200)],
    )

GNS here is computed with extra forward/backward passes on a captured batch
(the model is switched to eval() during estimation to exclude dropout/BN noise).
If you use gradient accumulation in a raw PyTorch loop and want GNS for *free*,
use traintools.gradnoise.GradientAccumulationGNS instead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from traintools.earlyguard import TrainGuard
from traintools.gradnoise import (
    GNSEstimator, GNSHistory, GNSResult,
    _classify, _unbiased_estimates, _gns_from_estimates,
)
from traintools.plasticity import PlasticityHistory, PlasticityProbe

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    class TrainerCallback:  # type: ignore
        pass
    TrainerControl = TrainerState = TrainingArguments = None


def _batch_size_of(batch: Dict[str, Any]) -> Optional[int]:
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return v.shape[0]
    return None


def _slice_batch(batch: Dict[str, Any], sl: slice) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v[sl] if isinstance(v, torch.Tensor) and v.dim() >= 1 else v
    return out


class TraintoolsCallback(TrainerCallback):
    """All-in-one traintools callback for the HuggingFace Trainer."""

    def __init__(
        self,
        gns_freq: int = 200,
        plasticity_freq: int = 200,
        gns_splits: int = 2,
        gns_ema_decay: float = 0.95,
        earlyguard: bool = True,
        min_improvement: float = 1e-4,
        patience_steps: int = 1000,
        horizon_steps: int = 500,
        verbose: bool = True,
    ) -> None:
        self.gns_freq = gns_freq
        self.plasticity_freq = plasticity_freq
        self.gns_splits = gns_splits
        self.verbose = verbose

        self._guard = TrainGuard(
            min_improvement=min_improvement,
            patience_steps=patience_steps,
            horizon_steps=horizon_steps,
        ) if earlyguard else None
        self._probe: Optional[PlasticityProbe] = None
        self._gns_estimator = GNSEstimator(decay=gns_ema_decay)
        self._gns_history = GNSHistory()
        self._plasticity_history = PlasticityHistory()

        self._model = None
        self._last_batch: Optional[Dict[str, Any]] = None
        self._capture_hook = None

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def on_train_begin(self, args, state, control, model=None, **kwargs) -> None:
        if model is None:
            return
        self._model = model
        self._probe = PlasticityProbe(model)
        # Capture the most recent batch fed to the model, for GNS estimation.
        def capture(_module, _args, kwargs_in):
            if kwargs_in:
                self._last_batch = {k: v for k, v in kwargs_in.items()}
        try:
            self._capture_hook = model.register_forward_pre_hook(capture, with_kwargs=True)
        except TypeError:
            # Older torch without with_kwargs — GNS capture unavailable, degrade gracefully.
            self._capture_hook = None

    def on_step_begin(self, args, state, control, **kwargs) -> None:
        if self._probe is not None and state.global_step % self.plasticity_freq == 0:
            self._probe.reset_buffers()

    def on_step_end(self, args, state, control, model=None, **kwargs) -> None:
        step = state.global_step
        model = model or self._model

        if self._probe is not None and step > 0 and step % self.plasticity_freq == 0:
            result = self._probe.measure(step=step)
            self._plasticity_history.record(result)
            if self.verbose:
                print(f"\n[traintools] {result}")
            if self._plasticity_history.is_degrading():
                print("[traintools] ! Plasticity degrading - consider reinitialising "
                      "dormant units or lowering LR.")

        if (model is not None and step > 0 and step % self.gns_freq == 0
                and self._last_batch is not None):
            if self._probe is not None:
                with self._probe.paused():
                    gns_result = self._estimate_gns(model, step)
            else:
                gns_result = self._estimate_gns(model, step)
            if gns_result is not None:
                self._gns_history.record(gns_result)
                if self.verbose:
                    print(f"\n[traintools:GNS]\n{gns_result}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        if self._guard is None or not metrics:
            return
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return
        self._guard.record(step=state.global_step, val_loss=val_loss)
        decision = self._guard.evaluate()
        if self.verbose:
            print(f"\n[traintools:TrainGuard] {decision}")
        if decision.should_stop:
            print("[traintools] Issuing early-stop signal.")
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs is None:
            return
        if self._plasticity_history.results:
            logs["traintools/plasticity_score"] = self._plasticity_history.scores[-1]
        if self._gns_history.results:
            logs["traintools/gns"] = self._gns_history.values[-1]

    def on_train_end(self, args, state, control, **kwargs) -> None:
        if self._capture_hook is not None:
            self._capture_hook.remove()
        if self._probe is not None:
            self._probe.remove_hooks()
        print("\n" + "=" * 60)
        print("[traintools] Training complete.")
        print(f"  {self._gns_history.summary()}")
        if self._plasticity_history.scores:
            print(f"  Final plasticity score: {self._plasticity_history.scores[-1]:.3f}")
        print("=" * 60)

    # ── GNS estimation via captured batch ─────────────────────────────────────

    def _estimate_gns(self, model, step: int) -> Optional[GNSResult]:
        batch = self._last_batch
        B = _batch_size_of(batch) if batch else None
        if not B or B < 2 * self.gns_splits:
            return None

        micro = B // self.gns_splits
        saved = {n: (p.grad.clone() if p.grad is not None else None)
                 for n, p in model.named_parameters()}
        was_training = model.training
        model.eval()
        micro_grads = []
        try:
            for i in range(self.gns_splits):
                sub = _slice_batch(batch, slice(i * micro, (i + 1) * micro))
                model.zero_grad(set_to_none=True)
                out = model(**sub)
                loss = out.loss if hasattr(out, "loss") else out["loss"]
                loss.backward()
                parts = [p.grad.detach().reshape(-1) for p in model.parameters()
                         if p.grad is not None]
                if not parts:
                    return None
                micro_grads.append(torch.cat(parts))
        except Exception as e:
            if self.verbose:
                print(f"[traintools:GNS] estimation skipped: {e}")
            return None
        finally:
            model.zero_grad(set_to_none=True)
            for n, p in model.named_parameters():
                if saved[n] is not None:
                    p.grad = saved[n]
            if was_training:
                model.train()

        g = torch.stack(micro_grads)
        return self._gns_estimator.update(g, micro, B, step=step)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def gns_history(self) -> GNSHistory:
        return self._gns_history

    @property
    def plasticity_history(self) -> PlasticityHistory:
        return self._plasticity_history

    @property
    def guard(self) -> Optional[TrainGuard]:
        return self._guard
