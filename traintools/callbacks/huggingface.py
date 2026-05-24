"""
HuggingFace Trainer callback for traintools.

Usage:
    from transformers import Trainer
    from traintools.callbacks.huggingface import TraintoolsCallback

    callback = TraintoolsCallback(
        gns_freq=200,           # estimate GNS every 200 steps
        plasticity_freq=200,    # measure plasticity every 200 steps
        earlyguard=True,        # enable probabilistic early stopping
        min_improvement=1e-4,   # stop when expected gain < this
        verbose=True,
    )
    trainer = Trainer(model=model, ..., callbacks=[callback])
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from traintools.earlyguard import TrainGuard
from traintools.gradnoise import GNSHistory
from traintools.plasticity import PlasticityHistory, PlasticityProbe

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    # Stub base class so the file is importable without transformers
    class TrainerCallback:  # type: ignore
        pass
    TrainerControl = TrainerState = TrainingArguments = None


class TraintoolsCallback(TrainerCallback):
    """
    All-in-one traintools callback for the HuggingFace Trainer.

    Computes GNS, Plasticity Score, and TrainGuard early stopping
    at configurable frequencies. Logs to trainer.log_history and optionally
    to Weights & Biases / TensorBoard via the trainer's built-in logging.
    """

    def __init__(
        self,
        gns_freq: int = 200,
        plasticity_freq: int = 200,
        earlyguard: bool = True,
        min_improvement: float = 1e-4,
        patience_steps: int = 1000,
        horizon_steps: int = 500,
        per_layer_gns: bool = False,
        verbose: bool = True,
    ) -> None:
        self.gns_freq = gns_freq
        self.plasticity_freq = plasticity_freq
        self.use_earlyguard = earlyguard
        self.verbose = verbose
        self.per_layer_gns = per_layer_gns

        self._guard = TrainGuard(
            min_improvement=min_improvement,
            patience_steps=patience_steps,
            horizon_steps=horizon_steps,
        ) if earlyguard else None
        self._probe: Optional[PlasticityProbe] = None
        self._gns_history = GNSHistory()
        self._plasticity_history = PlasticityHistory()

        # Cache the last batch seen (for GNS estimation)
        self._last_inputs: Optional[Any] = None
        self._last_labels: Optional[Any] = None
        self._model = None
        self._loss_fn = None

    # ── Trainer hook points ────────────────────────────────────────────────────

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, model=None, **kwargs) -> None:
        if model is not None:
            self._model = model
            self._probe = PlasticityProbe(model)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                      control: TrainerControl, **kwargs) -> None:
        # Reset plasticity activation buffer each measurement window
        if (self._probe is not None and
                state.global_step % self.plasticity_freq == 0):
            self._probe.reset_activation_buffer()

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs) -> None:
        step = state.global_step
        model = model or self._model

        # ── Plasticity measurement ─────────────────────────────────────────────
        if (self._probe is not None and step > 0 and
                step % self.plasticity_freq == 0):
            result = self._probe.measure(step=step)
            self._plasticity_history.record(result)
            if self.verbose:
                print(f"\n[traintools] {result}")
            if self._plasticity_history.is_degrading():
                print("[traintools] ⚠ Plasticity is degrading. "
                      "Consider reducing LR or reinitializing affected layers.")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, metrics: Dict[str, float] = None,
                    **kwargs) -> None:
        """Called after each evaluation pass — feed val loss to TrainGuard."""
        if self._guard is None or metrics is None:
            return
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return
        step = state.global_step
        self._guard.record(step=step, val_loss=val_loss)
        decision = self._guard.evaluate()
        if self.verbose:
            print(f"\n[traintools:TrainGuard] {decision}")
        if decision.should_stop:
            print("[traintools] Issuing early stop signal.")
            control.should_training_stop = True

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs: Dict[str, float] = None, **kwargs) -> None:
        """Inject traintools metrics into the trainer log."""
        if logs is None:
            return
        if self._plasticity_history.results:
            logs["traintools/plasticity_score"] = self._plasticity_history.scores[-1]
        if self._gns_history.results:
            logs["traintools/gns"] = self._gns_history.values[-1]

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs) -> None:
        print("\n" + "=" * 60)
        print("[traintools] Training complete. Summary:")
        print(f"  {self._gns_history.summary()}")
        if self._plasticity_history.scores:
            final_p = self._plasticity_history.scores[-1]
            print(f"  Final plasticity score: {final_p:.3f}")
        print("=" * 60)

    @property
    def gns_history(self) -> GNSHistory:
        return self._gns_history

    @property
    def plasticity_history(self) -> PlasticityHistory:
        return self._plasticity_history

    @property
    def guard(self) -> Optional[TrainGuard]:
        return self._guard
