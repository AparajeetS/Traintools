# Diagnose A Training Plateau

A plateau can have different causes:

1. Use `TrainGuard` to ask whether validation improvement is likely to continue.
2. Use `GradientHealthMonitor` to check whether updates vanished or exploded.
3. Use `PlasticityProbe` to inspect dormant units and feature effective rank.
4. Use `GradientConfusionMonitor` if examples may produce conflicting updates.

```python
from traintools import PlasticityProbe

probe = PlasticityProbe(model)
# Run representative forward passes.
result = probe.measure(step=step)
probe.reset_buffers()
```

Plasticity and neural-collapse geometry are descriptive. Do not treat either as
a universal model-quality score.
