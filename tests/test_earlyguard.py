"""Tests for TrainGuard early stopping oracle."""

import pytest

from traintools.earlyguard import TrainGuard, EarlyStopDecision


def _feed(guard, losses, step_size=50):
    for i, loss in enumerate(losses):
        guard.record(step=i * step_size, val_loss=loss)


def test_warmup_no_stop():
    guard = TrainGuard(warmup_records=5)
    _feed(guard, [0.9, 0.8, 0.7])  # only 3 records
    d = guard.evaluate()
    assert not d.should_stop
    assert "Warming up" in d.reason


def test_patience_triggers_stop():
    guard = TrainGuard(patience_steps=200, warmup_records=3)
    # Improve early, then plateau
    losses = [0.9, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6]
    _feed(guard, losses, step_size=100)
    d = guard.evaluate()
    assert d.should_stop
    assert "No improvement" in d.reason


def test_improving_run_continues():
    guard = TrainGuard(min_improvement=1e-4, patience_steps=10000, warmup_records=3)
    losses = [1.0, 0.8, 0.65, 0.55, 0.48, 0.43, 0.39, 0.36]
    _feed(guard, losses, step_size=100)
    d = guard.evaluate()
    assert not d.should_stop


def test_result_fields():
    guard = TrainGuard(warmup_records=3)
    _feed(guard, [0.9, 0.7, 0.6, 0.58, 0.57, 0.57])
    d = guard.evaluate()
    assert isinstance(d, EarlyStopDecision)
    assert d.current_loss == pytest.approx(0.57)
    assert isinstance(d.confidence_interval, tuple)
    assert len(d.confidence_interval) == 2


def test_reset():
    guard = TrainGuard(warmup_records=3)
    _feed(guard, [0.9, 0.8, 0.7, 0.7, 0.7])
    guard.reset()
    assert len(guard._steps) == 0
    d = guard.evaluate()
    assert "Warming up" in d.reason


def test_str_output():
    guard = TrainGuard(warmup_records=3)
    _feed(guard, [0.9, 0.7, 0.6, 0.59, 0.58])
    d = guard.evaluate()
    s = str(d)
    assert "current loss" in s
    assert "reason" in s
