"""
traintools — training diagnostics for PyTorch models.

Three tools:
  GNS (Gradient Noise Scale): are you using the right batch size?
  PlasticityProbe:            is your network losing the ability to learn?
  TrainGuard:                 should you stop training yet?

Quick start (raw PyTorch):
    from traintools.callbacks.pytorch import TraintoolsTracker
    tracker = TraintoolsTracker(model, loss_fn)
    # inside your training loop:
    tracker.step(step=global_step, inputs=x, targets=y, val_loss=val_loss)

Quick start (HuggingFace Trainer):
    from transformers import Trainer
    from traintools.callbacks.huggingface import TraintoolsCallback
    trainer = Trainer(model=model, ..., callbacks=[TraintoolsCallback()])
"""

from traintools.gradnoise import estimate_gns, GNSResult, GNSHistory
from traintools.plasticity import PlasticityProbe, PlasticityResult, PlasticityHistory
from traintools.earlyguard import TrainGuard, EarlyStopDecision

__version__ = "0.1.0"
__all__ = [
    "estimate_gns", "GNSResult", "GNSHistory",
    "PlasticityProbe", "PlasticityResult", "PlasticityHistory",
    "TrainGuard", "EarlyStopDecision",
]
