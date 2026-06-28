import torch
import torch.nn as nn

from traintools.confusion import (
    GradientConfusionMonitor,
    GradientConfusionResult,
    gradient_confusion_from_grads,
)


def test_gradient_confusion_from_grads_detects_conflict():
    grads = torch.tensor([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
    ])

    result = gradient_confusion_from_grads(grads, warn_negative_fraction=0.1, warn_min_cosine=-0.5)

    assert isinstance(result, GradientConfusionResult)
    assert result.min_cosine < 0
    assert result.negative_fraction > 0
    assert not result.ok


def test_gradient_confusion_from_aligned_grads_ok():
    grads = torch.tensor([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])

    result = gradient_confusion_from_grads(grads)

    assert result.min_cosine > 0.99
    assert result.negative_fraction == 0.0
    assert result.ok


def test_gradient_confusion_monitor_estimate_restores_grads():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(12, 4)
    y = torch.randint(0, 3, (12,))

    loss_fn(model(x), y).backward()
    before = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    result = GradientConfusionMonitor(n_splits=3).estimate(model, loss_fn, x, y, step=5)

    assert result.n_gradients == 3
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.allclose(p.grad, before[name])


def test_gradient_confusion_requires_two_gradients():
    try:
        gradient_confusion_from_grads(torch.randn(1, 4))
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:
        raise AssertionError("expected ValueError")
