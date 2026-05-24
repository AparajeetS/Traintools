"""Tests for PlasticityProbe (activation-based, Dohare/Sutton style)."""

import pytest
import torch
import torch.nn as nn

from traintools.plasticity import (
    PlasticityProbe, PlasticityHistory, PlasticityResult,
    activation_effective_rank,
)


@pytest.fixture
def model_and_batch():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    x = torch.randn(64, 16)
    y = torch.randint(0, 4, (64,))
    return model, x, y


def test_probe_returns_result(model_and_batch):
    model, x, _ = model_and_batch
    probe = PlasticityProbe(model)
    _ = model(x)
    result = probe.measure(step=1)
    assert isinstance(result, PlasticityResult)
    assert 0.0 <= result.global_score <= 1.0
    probe.remove_hooks()


def test_probe_healthy_network(model_and_batch):
    """A freshly initialised network with random inputs should be plastic."""
    model, x, _ = model_and_batch
    probe = PlasticityProbe(model)
    _ = model(x)
    result = probe.measure(step=10)
    assert result.global_score > 0.5
    probe.remove_hooks()


def test_probe_detects_dormant_units():
    """A ReLU layer that never activates (all-negative pre-activations) is dormant."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    # Force the first Linear to output strongly negative values → ReLU outputs 0
    with torch.no_grad():
        model[0].weight.fill_(0.0)
        model[0].bias.fill_(-100.0)   # every unit pinned off
    probe = PlasticityProbe(model)
    x = torch.randn(64, 16)
    _ = model(x)
    result = probe.measure(step=0)
    # The ReLU layer should be flagged as fully dormant → very low score
    relu_layer = [lr for lr in result.layers if "1" in lr.name]
    assert relu_layer, "expected a ReLU activation layer in results"
    assert relu_layer[0].dead_fraction > 0.9
    assert result.global_score < 0.3
    probe.remove_hooks()


def test_activation_effective_rank_full():
    """Isotropic random activations should have high effective rank."""
    torch.manual_seed(0)
    A = torch.randn(512, 64)
    er = activation_effective_rank(A)
    assert er > 0.5, f"random activations should have high erank, got {er:.3f}"


def test_activation_effective_rank_collapsed():
    """Rank-1 activations (all samples a multiple of one vector) → low erank."""
    torch.manual_seed(0)
    direction = torch.randn(1, 64)
    A = torch.randn(512, 1) @ direction   # every row colinear
    er = activation_effective_rank(A)
    assert er < 0.1, f"collapsed activations should have low erank, got {er:.3f}"


def test_plasticity_history():
    hist = PlasticityHistory()
    for i, score in enumerate([0.9, 0.85, 0.75, 0.6, 0.4]):
        hist.record(PlasticityResult(step=i * 100, global_score=score))
    assert hist.is_degrading(window=5, threshold=0.1)


def test_plasticity_stable():
    hist = PlasticityHistory()
    for i in range(5):
        hist.record(PlasticityResult(step=i * 100, global_score=0.95))
    assert not hist.is_degrading()


def test_reset_buffers_alias():
    """Both reset_buffers and the legacy alias should clear state."""
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    probe = PlasticityProbe(model)
    _ = model(torch.randn(16, 16))
    assert len(probe._buffers) > 0
    probe.reset_activation_buffer()   # legacy alias
    assert len(probe._buffers) == 0
    probe.remove_hooks()
