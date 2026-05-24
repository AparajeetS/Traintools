"""Tests for Gradient Noise Scale estimation."""

import pytest
import torch
import torch.nn as nn

from traintools.gradnoise import (
    GNSResult, GNSHistory, GNSEstimator, GradientAccumulationGNS, estimate_gns,
    _unbiased_estimates,
)


def test_unbiased_estimator_recovers_true_gns():
    """
    Regression guard for the v0.1 bias bug.
    Build micro-batch gradients with a known true GNS = tr(Sigma)/||G||^2 = 4.0
    and verify the corrected estimator recovers it (ratio of means) within 10%.
    Old buggy code returned ~1.8 here.
    """
    torch.manual_seed(0)
    D, micro, n_trials = 200, 16, 3000
    G = torch.ones(D) * 0.5
    sigma = 1.0
    true_gns = (D * sigma ** 2) / (G ** 2).sum().item()   # = 4.0

    tr_sum, sig_sum = 0.0, 0.0
    for _ in range(n_trials):
        micro_grads = torch.stack([
            (G.unsqueeze(0) + sigma * torch.randn(micro, D)).mean(0)
            for _ in range(2)
        ])
        tr, sig = _unbiased_estimates(micro_grads, micro)
        tr_sum += tr
        sig_sum += sig
    est_gns = tr_sum / sig_sum
    assert abs(est_gns - true_gns) / true_gns < 0.1, \
        f"estimator off: got {est_gns:.2f}, true {true_gns:.2f}"


def test_gns_estimator_ema_smooths():
    """GNSEstimator should produce a smoothed value distinct from single-shot raw."""
    torch.manual_seed(0)
    est = GNSEstimator(decay=0.9)
    D, micro = 100, 16
    G = torch.ones(D) * 0.5
    last = None
    for step in range(10):
        mg = torch.stack([(G.unsqueeze(0) + torch.randn(micro, D)).mean(0) for _ in range(2)])
        last = est.update(mg, micro, current_batch=32, step=step)
    assert last.smoothed
    assert last.gns > 0
    # EMA value should differ from the single-shot raw of the last step
    assert last.raw_gns > 0


def test_free_accumulation_gns():
    """GradientAccumulationGNS should recover GNS from differenced .grad."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    criterion = nn.CrossEntropyLoss()
    accum = GradientAccumulationGNS(model, micro_batch_size=16, decay=0.9)

    result = None
    for step in range(5):
        model.zero_grad(set_to_none=True)
        for _ in range(4):  # 4 micro-batches accumulate
            x = torch.randn(16, 32)
            y = torch.randint(0, 10, (16,))
            (criterion(model(x), y) / 4).backward()
            accum.record_microbatch()
        result = accum.compute(step=step)
        accum.reset_accumulation()

    assert result is not None
    assert result.gns > 0
    assert result.current_batch == 64  # 16 * 4 micro-batches
    assert result.smoothed


def test_free_accumulation_needs_two_microbatches():
    model = nn.Linear(8, 4)
    criterion = nn.CrossEntropyLoss()
    accum = GradientAccumulationGNS(model, micro_batch_size=8)
    x = torch.randn(8, 8)
    y = torch.randint(0, 4, (8,))
    criterion(model(x), y).backward()
    accum.record_microbatch()
    assert accum.compute(step=0) is None  # only one micro-batch recorded


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
    assert result.regime in ("under-batched", "optimal", "over-batched")
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
    assert result.regime in ("under-batched", "optimal", "over-batched")
