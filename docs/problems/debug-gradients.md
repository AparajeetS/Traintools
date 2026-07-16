# Debug NaNs, Exploding Gradients, And Slow SGD

Use `GradientHealthMonitor` for gradient finiteness, norms, clipping, and
update-to-weight ratios:

```python
from traintools import GradientHealthMonitor

monitor = GradientHealthMonitor(max_grad_norm=1.0)
loss.backward()
report = monitor.inspect(
    model,
    step=step,
    lr=optimizer.param_groups[0]["lr"],
)
if not report.ok:
    print(report)
optimizer.step()
```

Call it after `backward()` and before `optimizer.step()`.

Use `GradientConfusionMonitor` when gradients are finite but micro-batches may
be pulling in opposing directions. It adds gradient computations, so run it
periodically rather than every step.

Warnings are architecture-dependent evidence for investigation. They do not
identify the root cause automatically.
