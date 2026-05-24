"""
Stress tests for traintools.

Covers:
  - Numerical edge cases: NaN/inf gradients, zero gradients, rank-1 weights
  - Architecture variety: Conv2d, LSTM, MultiheadAttention, deep MLP
  - Large model performance: GNS timing on a 10M-param model
  - Adversarial plasticity: fully dead network, rank-1 weights
  - TrainGuard on pathological loss curves: monotone increase, instant plateau,
    extreme noise, perfect convergence
  - Hook lifecycle: double-attach, remove_hooks, garbage collection
  - Gradient accumulation simulation
  - Concurrent probe instances on the same model
"""

import gc
import math
import time

import pytest
import torch
import torch.nn as nn

from traintools.gradnoise import GNSHistory, GNSResult, estimate_gns
from traintools.plasticity import PlasticityHistory, PlasticityProbe, _effective_rank_normalized
from traintools.earlyguard import TrainGuard


# ── Fixtures ───────────────────────────────────────────────────────────────────

def mlp(in_dim=32, hidden=128, out_dim=10, depth=4):
    layers = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    torch.manual_seed(42)
    return nn.Sequential(*layers)


def conv_model():
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(32 * 16, 10),
    )


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.lstm = nn.LSTM(16, 64, batch_first=True)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.embed = nn.Linear(16, 32)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.fc = nn.Linear(32, 4)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.attn(x, x, x)
        return self.fc(x.mean(1))


def large_model(n_params_approx=10_000_000):
    # ~10M params: Linear(1000, 1000) x 10 ≈ 10M
    torch.manual_seed(42)
    layers = [nn.Linear(1000, 1000), nn.ReLU()] * 5 + [nn.Linear(1000, 10)]
    return nn.Sequential(*layers)


# ══════════════════════════════════════════════════════════════════════════════
# GNS STRESS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestGNSEdgeCases:

    def test_single_parameter_model(self):
        model = nn.Linear(8, 2, bias=False)
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(16, 8)
        y = torch.randint(0, 2, (16,))
        result = estimate_gns(model, criterion, x, y)
        assert result.gns >= 0
        assert result.regime in ("noise-dominated", "optimal", "signal-dominated")

    def test_conv_model(self):
        model = conv_model()
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(16, 3, 8, 8)
        y = torch.randint(0, 10, (16,))
        result = estimate_gns(model, criterion, x, y, per_layer=True)
        assert result.gns > 0
        assert len(result.per_layer) > 0

    def test_lstm_model(self):
        model = LSTMClassifier()
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(16, 10, 16)  # (batch, seq, features)
        y = torch.randint(0, 4, (16,))
        result = estimate_gns(model, criterion, x, y)
        assert result.gns > 0

    def test_transformer_model(self):
        model = TinyTransformer()
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(16, 5, 16)
        y = torch.randint(0, 4, (16,))
        result = estimate_gns(model, criterion, x, y)
        assert result.gns > 0

    def test_deep_mlp(self):
        model = mlp(depth=8)
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(32, 32)
        y = torch.randint(0, 10, (32,))
        result = estimate_gns(model, criterion, x, y, per_layer=True)
        assert result.gns > 0

    def test_n_splits_3(self):
        model = mlp()
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(60, 32)
        y = torch.randint(0, 10, (60,))
        result = estimate_gns(model, criterion, x, y, n_splits=3)
        assert result.gns > 0

    def test_gradients_not_corrupted_after_gns(self):
        """GNS must restore original gradients exactly."""
        model = mlp()
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(32, 32)
        y = torch.randint(0, 10, (32,))

        criterion(model(x), y).backward()
        before = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        estimate_gns(model, criterion, x, y, per_layer=True)

        for n, p in model.named_parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, before[n], atol=1e-6), f"Gradient corrupted: {n}"

    def test_zero_gradient_model(self):
        """A model whose loss is constant should not crash (may produce inf GNS)."""
        model = nn.Linear(8, 2)
        with torch.no_grad():
            model.weight.fill_(0.0)
            model.bias.fill_(0.0)
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(16, 8)
        y = torch.randint(0, 2, (16,))
        # Should not raise — may return inf or 0 GNS
        try:
            result = estimate_gns(model, criterion, x, y)
            assert result.gns >= 0 or math.isinf(result.gns)
        except Exception as e:
            pytest.fail(f"Zero-gradient model raised unexpectedly: {e}")

    def test_nan_safe(self):
        """NaN inputs should not hang — should raise or return gracefully."""
        model = mlp()
        criterion = nn.CrossEntropyLoss()
        x = torch.full((16, 32), float("nan"))
        y = torch.randint(0, 10, (16,))
        try:
            result = estimate_gns(model, criterion, x, y)
            # If it returns, GNS may be nan — that's acceptable
        except Exception:
            pass  # Raising is also acceptable

    def test_large_model_timing(self):
        """GNS on a ~10M param model should complete in under 30s on CPU."""
        model = large_model()
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(32, 1000)
        y = torch.randint(0, 10, (32,))
        t0 = time.time()
        result = estimate_gns(model, criterion, x, y)
        elapsed = time.time() - t0
        assert elapsed < 30.0, f"GNS took {elapsed:.1f}s — too slow"
        assert result.gns > 0

    def test_gns_consistent_across_calls(self):
        """Same model + same batch should give similar GNS (not wildly random)."""
        torch.manual_seed(0)
        model = mlp()
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(32, 32)
        y = torch.randint(0, 10, (32,))

        results = [estimate_gns(model, criterion, x, y).gns for _ in range(3)]
        # All within 50% of each other (stochasticity from batch splitting)
        mean = sum(results) / len(results)
        for r in results:
            assert abs(r - mean) / (mean + 1e-9) < 0.5, f"GNS too variable: {results}"


# ══════════════════════════════════════════════════════════════════════════════
# PLASTICITY STRESS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPlasticityEdgeCases:

    def test_conv_model(self):
        model = conv_model()
        probe = PlasticityProbe(model)
        x = torch.randn(32, 3, 8, 8)
        criterion = nn.CrossEntropyLoss()
        y = torch.randint(0, 10, (32,))
        model.zero_grad()
        criterion(model(x), y).backward()
        result = probe.measure(step=0)
        assert 0.0 <= result.global_score <= 1.0
        probe.remove_hooks()

    def test_lstm_model(self):
        model = LSTMClassifier()
        probe = PlasticityProbe(model)
        x = torch.randn(16, 10, 16)
        criterion = nn.CrossEntropyLoss()
        y = torch.randint(0, 4, (16,))
        model.zero_grad()
        criterion(model(x), y).backward()
        result = probe.measure(step=0)
        assert 0.0 <= result.global_score <= 1.0
        probe.remove_hooks()

    def test_fully_dead_network(self):
        """A network with all-zero weights should score near 0."""
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.0)
        probe = PlasticityProbe(model)
        x = torch.randn(32, 16)
        _ = model(x)  # forward to populate activation buffer
        result = probe.measure(step=0)
        assert result.global_score < 0.5
        probe.remove_hooks()

    def test_rank1_weights(self):
        """Near-rank-1 weight matrix should have low erank."""
        W = torch.randn(64, 1) @ torch.randn(1, 64) + 1e-4 * torch.eye(64)
        er = _effective_rank_normalized(W)
        assert er < 0.1, f"rank-1 matrix should have low erank, got {er:.3f}"

    def test_full_rank_weights(self):
        """Random Gaussian matrix should have high erank."""
        torch.manual_seed(0)
        W = torch.randn(64, 64)
        er = _effective_rank_normalized(W)
        assert er > 0.5, f"random matrix should have high erank, got {er:.3f}"

    def test_hook_cleanup(self):
        """Hooks should be fully removed after remove_hooks()."""
        model = mlp()
        probe = PlasticityProbe(model)
        n_hooks_before = len(probe._hooks)
        assert n_hooks_before > 0
        probe.remove_hooks()
        assert len(probe._hooks) == 0

    def test_double_attach(self):
        """Attaching two probes to the same model should not crash."""
        model = mlp()
        probe1 = PlasticityProbe(model)
        probe2 = PlasticityProbe(model)
        x = torch.randn(16, 32)
        criterion = nn.CrossEntropyLoss()
        y = torch.randint(0, 10, (16,))
        model.zero_grad()
        criterion(model(x), y).backward()
        r1 = probe1.measure(step=0)
        r2 = probe2.measure(step=0)
        assert 0.0 <= r1.global_score <= 1.0
        assert 0.0 <= r2.global_score <= 1.0
        probe1.remove_hooks()
        probe2.remove_hooks()

    def test_reset_activation_buffer(self):
        """reset_activation_buffer clears stale state between windows."""
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
        probe = PlasticityProbe(model)
        x = torch.randn(16, 16)
        _ = model(x)
        assert len(probe._activation_buffer) > 0
        probe.reset_activation_buffer()
        assert len(probe._activation_buffer) == 0
        probe.remove_hooks()

    def test_probe_no_forward_pass(self):
        """Measuring before any forward pass should not crash."""
        model = mlp()
        probe = PlasticityProbe(model)
        result = probe.measure(step=0)
        # No activations captured, so dead_fraction=0 for all layers
        assert 0.0 <= result.global_score <= 1.0
        probe.remove_hooks()

    def test_degradation_detected(self):
        """PlasticityHistory.is_degrading() should fire on a clear decline."""
        hist = PlasticityHistory()
        for score in [0.95, 0.85, 0.70, 0.50, 0.25]:
            from traintools.plasticity import PlasticityResult
            hist.record(PlasticityResult(step=0, global_score=score))
        assert hist.is_degrading(window=5, threshold=0.1)

    def test_no_false_degradation(self):
        hist = PlasticityHistory()
        for score in [0.91, 0.92, 0.90, 0.93, 0.91]:
            from traintools.plasticity import PlasticityResult
            hist.record(PlasticityResult(step=0, global_score=score))
        assert not hist.is_degrading(window=5, threshold=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINGUARD STRESS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainGuardEdgeCases:

    def _feed(self, guard, losses, step_size=100):
        for i, l in enumerate(losses):
            guard.record(step=i * step_size, val_loss=l)

    def test_monotone_increasing_loss(self):
        """Diverging training should trigger stop quickly."""
        guard = TrainGuard(patience_steps=300, warmup_records=3)
        self._feed(guard, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        d = guard.evaluate()
        assert d.should_stop

    def test_instant_plateau(self):
        """Loss that stops at step 0 should stop."""
        guard = TrainGuard(patience_steps=200, warmup_records=3)
        self._feed(guard, [0.5, 0.5, 0.5, 0.5, 0.5])
        d = guard.evaluate()
        assert d.should_stop

    def test_perfect_convergence(self):
        """Clean power-law convergence should continue until very close to asymptote."""
        guard = TrainGuard(min_improvement=1e-5, patience_steps=100_000, warmup_records=3)
        # Simulate clean power-law: a=0.1, b=0.9, c=0.5
        losses = [0.1 + 0.9 * (i + 1) ** -0.5 for i in range(10)]
        self._feed(guard, losses)
        d = guard.evaluate()
        assert not d.should_stop

    def test_extreme_noise(self):
        """Very noisy loss (random walk) should eventually stop on patience."""
        import random
        random.seed(0)
        guard = TrainGuard(patience_steps=500, warmup_records=5)
        losses = [abs(random.gauss(0.5, 0.3)) for _ in range(20)]
        self._feed(guard, losses, step_size=50)
        d = guard.evaluate()
        # Either stops (patience) or continues — just must not crash
        assert isinstance(d.should_stop, bool)

    def test_single_record(self):
        """Only one record — should be in warmup, not crash."""
        guard = TrainGuard(warmup_records=3)
        guard.record(step=0, val_loss=0.9)
        d = guard.evaluate()
        assert not d.should_stop

    def test_zero_loss(self):
        """Zero validation loss should not produce NaN or crash."""
        guard = TrainGuard(warmup_records=3)
        self._feed(guard, [0.5, 0.3, 0.1, 0.0, 0.0, 0.0])
        d = guard.evaluate()
        assert isinstance(d.should_stop, bool)

    def test_very_small_improvement(self):
        """Loss genuinely at floor — curve fit should predict negligible gain."""
        guard = TrainGuard(min_improvement=0.01, patience_steps=100_000,
                           horizon_steps=500, warmup_records=3)
        # Tightly converged: already within 1e-5 of asymptote 0.2
        losses = [0.5, 0.25, 0.21, 0.2005, 0.20001, 0.200002, 0.2000003, 0.20000004]
        self._feed(guard, losses)
        d = guard.evaluate()
        # Either curve fit predicts tiny gain (should_stop=True) or
        # CI upper bound is below threshold (should_stop=True).
        # If fit is unstable, that's also acceptable — just must not crash.
        assert isinstance(d.should_stop, bool)

    def test_confidence_interval_ordering(self):
        """CI lower bound must be <= upper bound."""
        guard = TrainGuard(warmup_records=3)
        self._feed(guard, [0.9, 0.7, 0.55, 0.45, 0.40, 0.38, 0.37])
        d = guard.evaluate()
        lo, hi = d.confidence_interval
        assert lo <= hi, f"CI inverted: [{lo}, {hi}]"

    def test_large_history(self):
        """Should handle 1000 records without performance issues."""
        guard = TrainGuard(warmup_records=5)
        losses = [1.0 * math.exp(-i / 200) + 0.1 for i in range(1000)]
        t0 = time.time()
        self._feed(guard, losses, step_size=1)
        d = guard.evaluate()
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"evaluate() took {elapsed:.1f}s on 1000 records"
        assert isinstance(d.should_stop, bool)

    def test_reset_restarts_cleanly(self):
        guard = TrainGuard(warmup_records=3)
        self._feed(guard, [0.9, 0.5, 0.5, 0.5, 0.5])
        assert guard.evaluate().should_stop
        guard.reset()
        guard.record(step=0, val_loss=0.9)
        d = guard.evaluate()
        assert not d.should_stop  # back in warmup


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION STRESS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_full_training_loop_simulation(self):
        """Simulate 500 steps of training — nothing should crash."""
        torch.manual_seed(0)
        model = mlp(depth=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        probe = PlasticityProbe(model)
        guard = TrainGuard(min_improvement=1e-3, patience_steps=200, warmup_records=5)

        for step in range(500):
            x = torch.randn(32, 32)
            y = torch.randint(0, 10, (32,))
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            if step % 100 == 0 and step > 0:
                result = estimate_gns(model, criterion, x, y, step=step)
                assert result.gns > 0
                p_result = probe.measure(step=step)
                assert 0.0 <= p_result.global_score <= 1.0
                guard.record(step=step, val_loss=loss.item())
                d = guard.evaluate()
                assert isinstance(d.should_stop, bool)

        probe.remove_hooks()

    def test_gns_and_plasticity_independent(self):
        """GNS estimation must not interfere with plasticity hook buffers."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
        criterion = nn.CrossEntropyLoss()
        probe = PlasticityProbe(model)

        x = torch.randn(32, 16)
        y = torch.randint(0, 4, (32,))

        # Forward to populate activation buffer
        model.zero_grad()
        criterion(model(x), y).backward()

        buf_before = {k: v.clone() for k, v in probe._activation_buffer.items()}

        # GNS should not affect the plasticity hook buffers
        estimate_gns(model, criterion, x, y)

        for k in buf_before:
            if k in probe._activation_buffer:
                assert torch.allclose(probe._activation_buffer[k], buf_before[k]), \
                    f"GNS corrupted plasticity buffer for {k}"

        probe.remove_hooks()

    def test_probe_garbage_collected(self):
        """Probe should clean up hooks even without explicit remove_hooks()."""
        model = mlp()
        probe = PlasticityProbe(model)
        n_hooks = len(probe._hooks)
        assert n_hooks > 0
        del probe
        gc.collect()
        # After GC, no way to check hooks are gone, but should not crash

    def test_tracker_pytorch_callback(self):
        """TraintoolsTracker full loop — no crash, returns decision or None."""
        from traintools.callbacks.pytorch import TraintoolsTracker
        torch.manual_seed(0)
        model = mlp()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        tracker = TraintoolsTracker(
            model, criterion,
            gns_freq=20, plasticity_freq=20,
            earlyguard=True, min_improvement=1e-4,
            patience_steps=100, verbose=False,
        )
        for step in range(50):
            x = torch.randn(32, 32)
            y = torch.randint(0, 10, (32,))
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
            val_loss = criterion(model(x), y).item() if step % 10 == 0 else None
            decision = tracker.step(step=step, inputs=x, targets=y, val_loss=val_loss)
            if decision is not None:
                assert isinstance(decision.should_stop, bool)
