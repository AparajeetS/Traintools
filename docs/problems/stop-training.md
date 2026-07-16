# Decide When To Stop Training

Use `TrainGuard` when independent validation loss has plateaued and the cost of
continuing matters.

```python
from traintools import TrainGuard

guard = TrainGuard(
    min_improvement=1e-4,
    patience_steps=1000,
    horizon_steps=500,
)

guard.record(step=step, val_loss=val_loss)
decision = guard.evaluate()
if decision.should_stop:
    break
```

For Hugging Face:

```python
from traintools.callbacks.huggingface import TraintoolsCallback

trainer = Trainer(model=model, ..., callbacks=[TraintoolsCallback()])
```

Do not feed training loss as though it were independent validation evidence.
Restart schedules, unfreezing, curriculum changes, and other regime shifts can
invalidate curve extrapolation.
