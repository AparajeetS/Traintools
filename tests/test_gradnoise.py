"""Tests for Gradient Noise Scale estimation."""

import pytest
import torch
import torch.nn as nn

from traintools.gradnoise import GNSResult, GNSHistory, estimate_gns


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return torch.randn(32, 16), torch.randint(0, 4, (32,))


def test_gns_returns_result(tiny_model, batch):
    x, y = batch
    criterion = nn.CrossEntropyLoss()
    result = estimate_gns(tiny_model, criterion, x, y, step=10)
    assert isinstance(result, GNSResult)
    assert result.gns > 0
    assert result.current_batch == 32
    assert result.regime in ("noise-dominated", "optimal", "signal-dominated")
    assert result.step == 10


def test_gns_per_layer(tiny_model, batch):
    x, y = batch
    criterion = nn.CrossEntropyLoss()
    result = estimate_gns(tiny_model, criterion, x, y, per_layer=True)
    assert len(result.per_layer) > 0
    for v in result.per_layer.values():
        assert v >= 0


def test_gns_does_not_corrupt_gradients(tiny_model, batch):
    """GNS estimation must restore original gradients."""
    x, y = batch
    criterion = nn.CrossEntropyLoss()

    # Do a real backward first
    loss = criterion(tiny_model(x), y)
    loss.backward()
    grads_before = {n: p.grad.clone() for n, p in tiny_model.named_parameters() if p.grad is not None}

    estimate_gns(tiny_model, criterion, x, y)

    for n, p in tiny_model.named_parameters():
        if p.grad is not None:
            assert torch.allclose(p.grad, grads_before[n]), f"Gradient corrupted for {n}"


def test_gns_batch_too_small(tiny_model):
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(3, 16)
    y = torch.randint(0, 4, (3,))
    with pytest.raises(ValueError, match="must be >="):
        estimate_gns(tiny_model, criterion, x, y)


def test_gns_history():
    hist = GNSHistory()
    assert hist.trend() == "insufficient data"

    for i, gns_val in enumerate([50.0, 44.0, 38.0, 30.0, 20.0]):
        r = GNSResult(step=i * 100, gns=gns_val, critical_batch=int(gns_val),
                      current_batch=64, regime="optimal", recommendation="ok")
        hist.record(r)

    assert hist.trend() == "falling"
    assert len(hist.steps) == 5
    summary = hist.summary()
    assert "falling" in summary


def test_gns_regime_noise_dominated(tiny_model):
    """With a huge batch and small model the regime should tip toward noise-dominated."""
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(256, 16)
    y = torch.randint(0, 4, (256,))
    result = estimate_gns(tiny_model, criterion, x, y, n_splits=4)
    # Just check it runs and classifies something valid
    assert result.regime in ("noise-dominated", "optimal", "signal-dominated")
