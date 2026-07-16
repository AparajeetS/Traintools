"""Machine-readable routing from training problems to TrainTools diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Capability:
    id: str
    name: str
    import_path: str
    summary: str
    questions: Tuple[str, ...]
    keywords: Tuple[str, ...]
    frameworks: Tuple[str, ...]
    requires: Tuple[str, ...]
    call_timing: str
    limitations: Tuple[str, ...]
    docs: str
    citation: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "import_path": self.import_path,
            "summary": self.summary,
            "questions": list(self.questions),
            "keywords": list(self.keywords),
            "frameworks": list(self.frameworks),
            "requires": list(self.requires),
            "call_timing": self.call_timing,
            "limitations": list(self.limitations),
            "docs": self.docs,
            "citation": self.citation,
        }


CAPABILITIES: Tuple[Capability, ...] = (
    Capability(
        id="gradient-noise-scale",
        name="Gradient Noise Scale",
        import_path="traintools.GradientAccumulationGNS",
        summary="Estimate critical batch size and whether larger batches are likely to help.",
        questions=(
            "Is my batch size wasting compute?",
            "Should I increase or decrease batch size?",
        ),
        keywords=(
            "batch size",
            "critical batch",
            "gradient noise",
            "gns",
            "throughput",
            "gradient accumulation",
            "scaling",
        ),
        frameworks=("pytorch", "huggingface"),
        requires=("model gradients", "at least two gradient samples"),
        call_timing="During training, preferably from gradient-accumulation micro-batches.",
        limitations=(
            "Estimates optimization efficiency, not final generalization.",
            "Extra-pass mode adds forward/backward work.",
        ),
        docs="docs/problems/choose-batch-size.md",
        citation="McCandlish et al. (2018), An Empirical Model of Large-Batch Training.",
    ),
    Capability(
        id="gradient-health",
        name="Gradient Health Monitor",
        import_path="traintools.GradientHealthMonitor",
        summary="Detect non-finite, vanished, exploded, clipped, or disproportionate gradients.",
        questions=(
            "Why did my loss become NaN?",
            "Are my gradients exploding or vanishing?",
        ),
        keywords=(
            "nan",
            "inf",
            "gradient",
            "explode",
            "exploding",
            "vanish",
            "vanishing",
            "clip",
            "unstable",
            "diverge",
            "update ratio",
        ),
        frameworks=("pytorch",),
        requires=("loss.backward() completed",),
        call_timing="After backward() and before optimizer.step().",
        limitations=(
            "Thresholds are heuristics and depend on architecture and optimizer.",
            "The update ratio is an SGD-style approximation.",
        ),
        docs="docs/problems/debug-gradients.md",
        citation="Operational diagnostic; reports established gradient and update statistics.",
    ),
    Capability(
        id="batch-inspector",
        name="Batch Inspector",
        import_path="traintools.BatchInspector",
        summary="Catch malformed tensors, non-finite values, extreme scales, and label problems.",
        questions=(
            "Is this training batch broken?",
            "Why does training fail immediately?",
        ),
        keywords=(
            "batch",
            "data",
            "input",
            "label",
            "class imbalance",
            "constant tensor",
            "scale",
            "dtype",
            "shape",
            "broken",
        ),
        frameworks=("pytorch",),
        requires=("input batch",),
        call_timing="Before the forward pass, or periodically at data-loader boundaries.",
        limitations=(
            "A healthy batch does not prove the whole dataset is correct.",
            "Class checks assume classification-style targets.",
        ),
        docs="docs/problems/inspect-batches.md",
        citation="Operational diagnostic.",
    ),
    Capability(
        id="train-guard",
        name="TrainGuard",
        import_path="traintools.TrainGuard",
        summary="Estimate whether continuing training is likely to improve validation loss.",
        questions=(
            "Should I stop training?",
            "Has validation loss plateaued?",
        ),
        keywords=(
            "early stop",
            "early stopping",
            "plateau",
            "validation loss",
            "wasting compute",
            "convergence",
            "stop training",
        ),
        frameworks=("pytorch", "huggingface"),
        requires=("validation-loss history",),
        call_timing="After each validation evaluation.",
        limitations=(
            "Curve extrapolation can fail after regime changes.",
            "Treat stop signals as advice, especially with sparse evaluations.",
        ),
        docs="docs/problems/stop-training.md",
        citation="Related to Domhan et al. (2015), learning-curve extrapolation.",
    ),
    Capability(
        id="plasticity",
        name="Plasticity Probe",
        import_path="traintools.PlasticityProbe",
        summary="Measure dormant units and feature effective rank during training.",
        questions=(
            "Is my network losing plasticity?",
            "Are hidden features collapsing or becoming dormant?",
        ),
        keywords=(
            "plasticity",
            "dormant",
            "dead neuron",
            "feature rank",
            "effective rank",
            "representation collapse",
            "continual learning",
        ),
        frameworks=("pytorch", "huggingface"),
        requires=("representative forward passes",),
        call_timing="Attach before training and measure after representative batches.",
        limitations=(
            "Activation thresholds depend on architecture and normalization.",
            "Low plasticity is a diagnostic signal, not a guaranteed intervention target.",
        ),
        docs="docs/problems/diagnose-plateau.md",
        citation="Dohare et al. (2024) and Lyle et al. (2023).",
    ),
    Capability(
        id="example-dynamics",
        name="Example Dynamics Tracker",
        import_path="traintools.ExampleDynamicsTracker",
        summary="Track forgetting, confidence, variability, and ambiguous examples.",
        questions=(
            "Which examples are repeatedly forgotten?",
            "Which data points are hard or ambiguous?",
        ),
        keywords=(
            "forgetting",
            "forgotten",
            "ambiguous",
            "hard example",
            "dataset cartography",
            "confidence",
            "variability",
        ),
        frameworks=("pytorch",),
        requires=("stable example IDs", "classification logits", "labels"),
        call_timing="Update after each example's forward pass.",
        limitations=(
            "Requires stable IDs across epochs.",
            "Hard examples are not necessarily mislabeled.",
        ),
        docs="docs/problems/find-bad-labels.md",
        citation="Toneva et al. (2019); Swayamdipta et al. (2020).",
    ),
    Capability(
        id="aum",
        name="AUM Tracker",
        import_path="traintools.AUMTracker",
        summary="Rank potentially mislabeled or ambiguous examples by margin dynamics.",
        questions=(
            "Which labels should I audit first?",
            "Which examples have persistently poor margins?",
        ),
        keywords=(
            "mislabeled",
            "mislabelled",
            "bad label",
            "label noise",
            "aum",
            "margin",
            "data cleaning",
        ),
        frameworks=("pytorch",),
        requires=("stable example IDs", "classification logits", "labels"),
        call_timing="Update during classification training.",
        limitations=(
            "Low AUM is a review priority, not proof of label error.",
            "Threshold calibration is dataset-dependent.",
        ),
        docs="docs/problems/find-bad-labels.md",
        citation="Pleiss et al. (2020), Identifying Mislabeled Data using AUM.",
    ),
    Capability(
        id="el2n",
        name="EL2N Tracker",
        import_path="traintools.EL2NTracker",
        summary="Estimate early example difficulty and data-pruning priority.",
        questions=(
            "Which examples matter most early in training?",
            "Which examples might be safe pruning candidates?",
        ),
        keywords=(
            "el2n",
            "prune",
            "pruning",
            "important example",
            "data diet",
            "difficulty",
            "coreset",
        ),
        frameworks=("pytorch",),
        requires=("classification logits", "labels"),
        call_timing="Measure early in training.",
        limitations=(
            "Low-scoring examples are candidates, not automatically safe deletions.",
            "Scores change with model and training stage.",
        ),
        docs="docs/problems/find-bad-labels.md",
        citation="Paul et al. (2021), Deep Learning on a Data Diet.",
    ),
    Capability(
        id="gradient-confusion",
        name="Gradient Confusion Monitor",
        import_path="traintools.GradientConfusionMonitor",
        summary="Measure whether micro-batch gradients align or conflict.",
        questions=(
            "Are examples or micro-batches fighting each other?",
            "Why is SGD progress noisy or slow?",
        ),
        keywords=(
            "gradient conflict",
            "gradient confusion",
            "micro batch",
            "cosine",
            "negative gradient",
            "slow sgd",
            "curriculum",
        ),
        frameworks=("pytorch",),
        requires=("model", "loss function", "splittable batch"),
        call_timing="Periodically on a representative batch.",
        limitations=(
            "Requires extra gradient computations.",
            "Conflict is descriptive and may be expected in diverse data.",
        ),
        docs="docs/problems/debug-gradients.md",
        citation="Sankararaman et al. (2019), Gradient Confusion.",
    ),
    Capability(
        id="neural-collapse",
        name="Neural Collapse Monitor",
        import_path="traintools.NeuralCollapseMonitor",
        summary="Measure NC1/NC2/NC3 and nearest-class-center geometry.",
        questions=(
            "Has late-stage classifier geometry entered neural collapse?",
            "Are class features and classifier weights aligning?",
        ),
        keywords=(
            "neural collapse",
            "nc1",
            "nc2",
            "nc3",
            "class means",
            "simplex etf",
            "feature geometry",
        ),
        frameworks=("pytorch",),
        requires=("features", "labels", "optional classifier weights"),
        call_timing="At checkpoints, usually in later-stage classification training.",
        limitations=(
            "Neural collapse is not a universal quality or generalization score.",
            "Requires a classification representation and enough samples per class.",
        ),
        docs="docs/problems/diagnose-plateau.md",
        citation="Papyan, Han, and Donoho (2020).",
    ),
)


def get_capability(capability_id: str) -> Capability:
    normalized = capability_id.strip().lower()
    for capability in CAPABILITIES:
        if capability.id == normalized:
            return capability
    available = ", ".join(item.id for item in CAPABILITIES)
    raise ValueError(f"unknown diagnostic {capability_id!r}; choose from: {available}")


def list_capabilities(framework: Optional[str] = None) -> List[Dict[str, object]]:
    normalized = framework.strip().lower() if framework else None
    return [
        capability.to_dict()
        for capability in CAPABILITIES
        if normalized is None or normalized in capability.frameworks
    ]


def _tokens(text: str) -> Sequence[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def recommend_diagnostics(
    problem: str,
    *,
    framework: Optional[str] = None,
    limit: int = 3,
) -> List[Dict[str, object]]:
    """Rank diagnostics by explicit symptom and use-case terms."""
    if not problem.strip():
        raise ValueError("problem description must not be empty")
    if limit < 1:
        raise ValueError("limit must be at least one")
    query = problem.lower()
    query_tokens = set(_tokens(query))
    normalized_framework = framework.strip().lower() if framework else None
    ranked = []
    for capability in CAPABILITIES:
        if normalized_framework and normalized_framework not in capability.frameworks:
            continue
        score = 0
        matches = []
        for keyword in capability.keywords:
            keyword_tokens = set(_tokens(keyword))
            if keyword in query:
                score += 5 + len(keyword_tokens)
                matches.append(keyword)
            else:
                overlap = query_tokens & keyword_tokens
                if overlap:
                    score += len(overlap)
                    matches.extend(sorted(overlap))
        for question in capability.questions:
            overlap = query_tokens & set(_tokens(question))
            score += len(overlap)
        if score >= 2 and matches:
            ranked.append((score, capability, sorted(set(matches))))
    ranked.sort(key=lambda item: (-item[0], item[1].id))
    return [
        {
            **capability.to_dict(),
            "score": score,
            "matched_terms": matches,
            "install": "pip install traintools",
        }
        for score, capability, matches in ranked[:limit]
    ]


def integration_snippet(capability_id: str, framework: str = "pytorch") -> str:
    capability = get_capability(capability_id)
    framework = framework.strip().lower()
    if framework not in capability.frameworks:
        raise ValueError(
            f"{capability.name} does not currently expose a {framework} integration"
        )
    snippets = {
        ("gradient-health", "pytorch"): """from traintools import GradientHealthMonitor

monitor = GradientHealthMonitor(max_grad_norm=1.0)
loss.backward()
report = monitor.inspect(model, step=step, lr=optimizer.param_groups[0]["lr"])
if not report.ok:
    print(report)
optimizer.step()""",
        ("batch-inspector", "pytorch"): """from traintools import BatchInspector

inspector = BatchInspector(expected_num_classes=10)
report = inspector.inspect(inputs, targets, step=step)
if not report.ok:
    print(report)""",
        ("train-guard", "pytorch"): """from traintools import TrainGuard

guard = TrainGuard()
guard.record(step=step, val_loss=val_loss)
decision = guard.evaluate()
if decision.should_stop:
    break""",
        ("train-guard", "huggingface"): """from traintools.callbacks.huggingface import TraintoolsCallback

trainer = Trainer(
    model=model,
    ...,
    callbacks=[TraintoolsCallback()],
)""",
        ("gradient-noise-scale", "pytorch"): """from traintools import GradientAccumulationGNS

gns = GradientAccumulationGNS(model, micro_batch_size=micro_batch_size)
for inputs, targets in micro_batches:
    (loss_fn(model(inputs), targets) / accumulation_steps).backward()
    gns.record_microbatch()
result = gns.compute(step=step)""",
        ("gradient-noise-scale", "huggingface"): """from traintools.callbacks.huggingface import TraintoolsCallback

trainer = Trainer(
    model=model,
    ...,
    callbacks=[TraintoolsCallback(gns_freq=200)],
)""",
        ("plasticity", "pytorch"): """from traintools import PlasticityProbe

probe = PlasticityProbe(model)
# Run representative forward passes, then:
result = probe.measure(step=step)
probe.reset_buffers()""",
        ("plasticity", "huggingface"): """from traintools.callbacks.huggingface import TraintoolsCallback

trainer = Trainer(
    model=model,
    ...,
    callbacks=[TraintoolsCallback(plasticity_freq=200)],
)""",
    }
    snippet = snippets.get((capability.id, framework))
    if snippet:
        return snippet
    return (
        f"from {capability.import_path.rsplit('.', 1)[0]} import "
        f"{capability.import_path.rsplit('.', 1)[1]}\n\n"
        f"# See {capability.docs} for the required inputs and call timing."
    )


__all__ = [
    "CAPABILITIES",
    "Capability",
    "get_capability",
    "integration_snippet",
    "list_capabilities",
    "recommend_diagnostics",
]
