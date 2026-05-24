"""
TrainGuard — probabilistic early stopping oracle.

Fits a curve to the validation loss history and predicts:
  - Whether the model is still improving meaningfully
  - Estimated steps until improvement falls below a threshold
  - Probability that continuing for N more steps yields gain > epsilon

Two curve models are available:
  'power'  : loss(t) = a + b * t^(-c)    — asymptotic power law (most common)
  'exp'    : loss(t) = a + b * exp(-c*t) — exponential decay

Fitting uses scipy.optimize.curve_fit with bootstrap uncertainty.
Falls back to linear extrapolation if scipy is unavailable.
"""

from __future__ import annotations

import math
import statistics
import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch


@dataclass
class EarlyStopDecision:
    step: int
    current_loss: float
    predicted_final_loss: float
    predicted_improvement: float           # loss reduction expected if training continues
    confidence_interval: Tuple[float, float]  # 90% CI on improvement
    steps_to_plateau: Optional[int]        # estimated steps until improvement < threshold
    should_stop: bool
    reason: str

    def __str__(self) -> str:
        ci_lo, ci_hi = self.confidence_interval
        lines = [
            f"[step {self.step}] {'STOP' if self.should_stop else 'CONTINUE'}",
            f"  current loss: {self.current_loss:.4f}",
            f"  predicted final: {self.predicted_final_loss:.4f}",
            f"  expected improvement: {self.predicted_improvement:.4f} "
            f"(90% CI: [{ci_lo:.4f}, {ci_hi:.4f}])",
        ]
        if self.steps_to_plateau is not None:
            lines.append(f"  estimated plateau at step: {self.step + self.steps_to_plateau}")
        lines.append(f"  reason: {self.reason}")
        return "\n".join(lines)


def _power_law(t, a, b, c):
    return a + b * t ** (-c)


def _exp_decay(t, a, b, c):
    return a + b * math.exp(-c * t) if isinstance(t, float) else a + b * torch.exp(-c * t)


def _fit_curve(
    steps: List[int],
    losses: List[float],
    model: str,
) -> Optional[tuple]:
    """
    Returns (params, param_std) or None on failure.
    Requires scipy.
    """
    try:
        import numpy as np
        from scipy.optimize import curve_fit

        t = np.array(steps, dtype=float)
        y = np.array(losses, dtype=float)

        if model == "power":
            # Initial guess: a=min(y), b=range(y), c=1
            p0 = [min(y) * 0.9, max(y) - min(y), 1.0]
            bounds = ([0, 0, 0.01], [max(y) * 2, max(y) * 10, 10])
            popt, pcov = curve_fit(_power_law_np, t, y, p0=p0, bounds=bounds, maxfev=5000)
        else:
            p0 = [min(y) * 0.9, max(y) - min(y), 0.001]
            bounds = ([0, 0, 1e-6], [max(y) * 2, max(y) * 10, 1])
            popt, pcov = curve_fit(_exp_decay_np, t, y, p0=p0, bounds=bounds, maxfev=5000)

        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    except Exception:
        return None


def _power_law_np(t, a, b, c):
    import numpy as np
    with np.errstate(over="ignore", invalid="ignore"):
        return a + b * np.power(np.maximum(t, 1e-6), -c)


def _exp_decay_np(t, a, b, c):
    import numpy as np
    return a + b * np.exp(-c * t)


def _linear_extrapolate(
    steps: List[int],
    losses: List[float],
    horizon: int,
) -> Tuple[float, Tuple[float, float]]:
    """Fallback: linear fit on last 20% of history."""
    n = max(2, len(steps) // 5)
    ts = steps[-n:]
    ls = losses[-n:]

    t_mean = sum(ts) / len(ts)
    l_mean = sum(ls) / len(ls)
    cov = sum((t - t_mean) * (l - l_mean) for t, l in zip(ts, ls))
    var = sum((t - t_mean) ** 2 for t in ts) + 1e-12
    slope = cov / var
    intercept = l_mean - slope * t_mean

    t_future = steps[-1] + horizon
    predicted = intercept + slope * t_future

    residuals = [l - (intercept + slope * t) for t, l in zip(ts, ls)]
    std = statistics.stdev(residuals) if len(residuals) > 1 else 0.0
    return predicted, (predicted - 1.65 * std, predicted + 1.65 * std)


class TrainGuard:
    """
    Probabilistic early stopping oracle.

    Usage:
        guard = TrainGuard(min_improvement=1e-4, patience_steps=500)
        # In validation loop:
        guard.record(step=step, val_loss=val_loss)
        decision = guard.evaluate()
        if decision.should_stop:
            print(decision)
            break
    """

    def __init__(
        self,
        min_improvement: float = 1e-4,
        patience_steps: int = 1000,
        horizon_steps: int = 500,
        curve_model: Literal["power", "exp", "auto"] = "auto",
        warmup_records: int = 5,
    ) -> None:
        """
        Args:
            min_improvement:  minimum expected absolute loss reduction to justify continuing
            patience_steps:   if no improvement in this many steps, stop
            horizon_steps:    how far ahead to project the loss curve
            curve_model:      'power', 'exp', or 'auto' (tries both, picks lower residual)
            warmup_records:   minimum number of val loss records before issuing STOP decisions
        """
        self.min_improvement = min_improvement
        self.patience_steps = patience_steps
        self.horizon_steps = horizon_steps
        self.curve_model = curve_model
        self.warmup_records = warmup_records

        self._steps: List[int] = []
        self._losses: List[float] = []
        self._best_loss: float = float("inf")
        self._best_step: int = 0

    def record(self, step: int, val_loss: float) -> None:
        self._steps.append(step)
        self._losses.append(val_loss)
        if val_loss < self._best_loss:
            self._best_loss = val_loss
            self._best_step = step

    def evaluate(self) -> EarlyStopDecision:
        step = self._steps[-1] if self._steps else 0
        current_loss = self._losses[-1] if self._losses else float("inf")

        if len(self._steps) < self.warmup_records:
            return EarlyStopDecision(
                step=step,
                current_loss=current_loss,
                predicted_final_loss=current_loss,
                predicted_improvement=float("inf"),
                confidence_interval=(float("-inf"), float("inf")),
                steps_to_plateau=None,
                should_stop=False,
                reason=f"Warming up ({len(self._steps)}/{self.warmup_records} records).",
            )

        # Patience check (cheap, always wins over curve fitting)
        steps_since_best = step - self._best_step
        if steps_since_best > self.patience_steps:
            return EarlyStopDecision(
                step=step,
                current_loss=current_loss,
                predicted_final_loss=self._best_loss,
                predicted_improvement=0.0,
                confidence_interval=(0.0, 0.0),
                steps_to_plateau=0,
                should_stop=True,
                reason=(f"No improvement in {steps_since_best} steps "
                        f"(best={self._best_loss:.4f} at step {self._best_step})."),
            )

        # Curve fitting
        fit = self._fit_best_curve()

        if fit is not None:
            predicted_final, ci = fit
            improvement = current_loss - predicted_final
            steps_to_plateau = self._estimate_plateau(predicted_final)
        else:
            predicted_final, ci = _linear_extrapolate(
                self._steps, self._losses, self.horizon_steps
            )
            improvement = current_loss - predicted_final
            steps_to_plateau = None

        if improvement < 0:
            # Predicted final loss is higher than current — curve fit is unstable.
            # Don't fire a false STOP; fall back to patience check only.
            should_stop = False
            reason = (f"Curve fit unstable (predicted improvement {improvement:.6f} < 0). "
                      f"Defaulting to patience check — no stop signal.")
        elif improvement <= self.min_improvement:
            should_stop = True
            reason = (f"Predicted improvement over next {self.horizon_steps} steps is "
                      f"{improvement:.6f} < threshold {self.min_improvement:.6f}.")
        elif ci[1] <= self.min_improvement:
            should_stop = True
            reason = (f"Upper 90% CI on improvement ({ci[1]:.6f}) < threshold. "
                      f"Unlikely to improve meaningfully.")
        else:
            should_stop = False
            reason = (f"Expected improvement {improvement:.4f} > threshold {self.min_improvement}. "
                      f"Continue training.")

        return EarlyStopDecision(
            step=step,
            current_loss=current_loss,
            predicted_final_loss=predicted_final,
            predicted_improvement=improvement,
            confidence_interval=ci,
            steps_to_plateau=steps_to_plateau,
            should_stop=should_stop,
            reason=reason,
        )

    def _fit_best_curve(self) -> Optional[Tuple[float, Tuple[float, float]]]:
        """Try power and exp fits, return (predicted_final, ci_90) for the better one."""
        try:
            import numpy as np
        except ImportError:
            return None

        models = (["power", "exp"] if self.curve_model == "auto"
                  else [self.curve_model])
        best_residual = float("inf")
        best_result = None

        for model in models:
            fit = _fit_curve(self._steps, self._losses, model)
            if fit is None:
                continue
            popt, perr = fit

            t = np.array(self._steps)
            y = np.array(self._losses)
            if model == "power":
                y_pred = _power_law_np(t, *popt)
                fn = _power_law_np
            else:
                y_pred = _exp_decay_np(t, *popt)
                fn = _exp_decay_np

            residual = float(np.mean((y - y_pred) ** 2))
            if residual < best_residual:
                best_residual = residual
                t_future = self._steps[-1] + self.horizon_steps
                pred = float(fn(t_future, *popt))
                # Propagate uncertainty: Monte Carlo with 200 samples
                rng_samples = []
                for _ in range(200):
                    p_sample = popt + np.random.randn(len(popt)) * perr
                    rng_samples.append(float(fn(t_future, *p_sample)))
                ci_lo = float(np.percentile(rng_samples, 5))
                ci_hi = float(np.percentile(rng_samples, 95))
                best_result = (pred, (ci_lo, ci_hi))

        return best_result

    def _estimate_plateau(self, asymptote: float) -> Optional[int]:
        """
        Estimate how many more steps until loss is within 1e-4 of asymptote.
        Only works for power-law model where we can invert analytically.
        """
        fit = _fit_curve(self._steps, self._losses, "power")
        if fit is None:
            return None
        popt, _ = fit
        a, b, c = popt
        gap = self._losses[-1] - asymptote
        target_gap = 1e-4
        if gap <= target_gap or c <= 0 or b <= 0:
            return 0
        try:
            import numpy as np
            if c < 0.01:
                return None  # exponent too shallow to give a meaningful estimate
            # a + b*t^(-c) = asymptote + target_gap → t = (b/target_gap)^(1/c)
            t_plateau = (b / target_gap) ** (1.0 / c)
            delta = max(0, int(t_plateau) - self._steps[-1])
            # Cap at 100x the current training horizon to avoid numeric blowup
            horizon = max(1, self._steps[-1])
            return min(delta, 100 * horizon)
        except Exception:
            return None

    def reset(self) -> None:
        self._steps.clear()
        self._losses.clear()
        self._best_loss = float("inf")
        self._best_step = 0

    def plot(self, ax=None, extra_steps: int = 500):
        import matplotlib.pyplot as plt
        import numpy as np
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))

        ax.plot(self._steps, self._losses, marker="o", linewidth=1.5,
                markersize=4, label="val loss", color="steelblue")

        fit = _fit_curve(self._steps, self._losses, "power") or _fit_curve(self._steps, self._losses, "exp")
        if fit is not None:
            popt, perr = fit
            t_fut = np.linspace(self._steps[-1], self._steps[-1] + extra_steps, 100)
            # Try power first
            y_fut = _power_law_np(t_fut, *popt)
            ax.plot(t_fut, y_fut, linestyle="--", color="orange", label="projected")
            ax.axhline(popt[0], linestyle=":", color="gray", linewidth=0.8, label=f"asymptote≈{popt[0]:.4f}")

        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation loss")
        ax.set_title("TrainGuard: Loss Projection")
        ax.legend(fontsize=8)
        return ax
