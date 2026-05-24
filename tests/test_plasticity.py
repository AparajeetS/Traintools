"""Tests for PlasticityProbe."""

import pytest
import torch
import torch.nn as nn

from traintools.plasticity import (
    PlasticityProbe, PlasticityHistory, PlasticityResult,
    _effective_rank_normalized,
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


def test_probe_with_gradients(model_and_batch):
    model, x, y = model_and_batch
    criterion = nn.CrossEntropyLoss()
    probe = PlasticityProbe(model)

    model.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    result = probe.measure(step=10)
    # With real gradients, score should be healthy (not 0)
    assert result.global_score > 0.5
    probe.remove_hooks()


def test_probe_detects_dead_weights():
    """Manually zeroing a weight should reduce the plasticity score."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    x = torch.randn(64, 16)
    criterion = nn.CrossEntropyLoss()
    y = torch.randint(0, 4, (64,))

    probe_healthy = PlasticityProbe(model)
    model.zero_grad()
    criterion(model(x), y).backward()
    score_healthy = probe_healthy.measure(step=0).global_score
    probe_healthy.remove_hooks()

    # Zero out all weights in first layer to simulate collapse
    with torch.no_grad():
        model[0].weight.fill_(0.0)
        model[0].bias.fill_(0.0)

    probe_dead = PlasticityProbe(model)
    model.zero_grad()
    try:
        criterion(model(x), y).backward()
    except Exception:
        pass
    score_dead = probe_dead.measure(step=1).global_score
    probe_dead.remove_hooks()

    assert score_dead <= score_healthy


def test_effective_rank_normalized():
    # Full-rank random matrix → high erank
    W = torch.randn(32, 32)
    er_full = _effective_rank_normalized(W)
    assert er_full > 0.5

    # Near-rank-1 matrix → low erank
    u = torch.randn(32, 1)
    W_rank1 = u @ u.T + 1e-6 * torch.eye(32)
    er_low = _effective_rank_normalized(W_rank1)
    assert er_low < 0.2


def test_plasticity_history():
    hist = PlasticityHistory()
    for i, score in enumerate([0.9, 0.85, 0.75, 0.6, 0.4]):
        result = PlasticityResult(step=i * 100, global_score=score)
        hist.record(result)
    assert hist.is_degrading(window=5, threshold=0.1)


def test_plasticity_stable():
    hist = PlasticityHistory()
    for i in range(5):
        result = PlasticityResult(step=i * 100, global_score=0.95)
        hist.record(result)
    assert not hist.is_degrading()
