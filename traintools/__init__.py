"""
traintools — training diagnostics for PyTorch models.

Included diagnostics:
  GNS (Gradient Noise Scale): are you using the right batch size?
  PlasticityProbe:            is your network losing the ability to learn?
  TrainGuard:                 should you stop training yet?
  BatchInspector:             is the current batch healthy?
  GradientHealthMonitor:      are gradients and update sizes sane?
  ExampleDynamicsTracker:     which examples are forgotten, hard, or ambiguous?
  GradientConfusionMonitor:   do micro-batch gradients fight each other?
  AUMTracker:                 which labels look suspicious by margin dynamics?
  EL2NTracker:                which examples are important early in training?
  NeuralCollapseMonitor:      has the classifier entered neural-collapse geometry?

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

from traintools.gradnoise import (
    estimate_gns, GNSResult, GNSHistory, GNSEstimator, GradientAccumulationGNS,
)
from traintools.plasticity import (
    PlasticityProbe, PlasticityResult, PlasticityHistory, activation_effective_rank,
)
from traintools.earlyguard import TrainGuard, EarlyStopDecision
from traintools.batch import BatchInspector, BatchReport, TensorStats
from traintools.gradhealth import (
    GradientHealthMonitor, GradientHealthResult, LayerGradientStats,
)
from traintools.dynamics import (
    ExampleDynamicsTracker, ExampleDynamics, DynamicsSummary,
)
from traintools.confusion import (
    GradientConfusionMonitor, GradientConfusionResult, gradient_confusion_from_grads,
)
from traintools.aum import AUMTracker, AUMExample, AUMSummary
from traintools.importance import EL2NTracker, EL2NExample, EL2NSummary, el2n_scores
from traintools.collapse import NeuralCollapseMonitor, NeuralCollapseResult
from traintools.capabilities import (
    get_capability, integration_snippet, list_capabilities, recommend_diagnostics,
)
from traintools.reporting import report_envelope, to_jsonable, write_json_report

__version__ = "0.6.2"
__all__ = [
    "estimate_gns", "GNSResult", "GNSHistory", "GNSEstimator", "GradientAccumulationGNS",
    "PlasticityProbe", "PlasticityResult", "PlasticityHistory", "activation_effective_rank",
    "TrainGuard", "EarlyStopDecision",
    "BatchInspector", "BatchReport", "TensorStats",
    "GradientHealthMonitor", "GradientHealthResult", "LayerGradientStats",
    "ExampleDynamicsTracker", "ExampleDynamics", "DynamicsSummary",
    "GradientConfusionMonitor", "GradientConfusionResult", "gradient_confusion_from_grads",
    "AUMTracker", "AUMExample", "AUMSummary",
    "EL2NTracker", "EL2NExample", "EL2NSummary", "el2n_scores",
    "NeuralCollapseMonitor", "NeuralCollapseResult",
    "get_capability", "integration_snippet", "list_capabilities", "recommend_diagnostics",
    "report_envelope", "to_jsonable", "write_json_report",
]
