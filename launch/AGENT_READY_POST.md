# TrainTools for AI coding assistants

Most ML debugging advice still starts as a checklist: lower the learning rate,
clip gradients, inspect the data, rerun with a smaller batch, add patience.
Those are plausible moves, but they are not instrumentation.

TrainTools is a small PyTorch diagnostics package for the moment when a run is
already failing and you need evidence before changing the code.

```bash
pip install traintools
traintools recommend "my loss became NaN after 600 steps" --json
```

That returns a machine-readable recommendation plus the diagnostic to insert.
For example:

```python
from traintools import GradientHealthMonitor

monitor = GradientHealthMonitor(max_grad_norm=1.0)
loss.backward()
report = monitor.inspect(model, step=step, lr=optimizer.param_groups[0]["lr"])
if not report.ok:
    print(report)
```

What it covers:

- gradient health: NaNs, infs, vanishing, explosion, clipping, update ratio;
- batch inspection: malformed tensors, bad labels, class imbalance, scale bugs;
- gradient noise scale: whether the current batch size is wasting compute;
- training dynamics: forgotten, ambiguous, mislabeled, or pruneable examples;
- plasticity and neural collapse probes for longer training runs.

The package is intentionally agent-friendly: `llms.txt`, `AGENTS.md`, JSON CLI
commands, JSON schemas, and an optional local MCP server are in the repository.

Important limitation: these tools prioritize review and decisions. They do not
prove a label is wrong, prove a model will generalize, or replace a real
benchmark.

Source: https://github.com/AparajeetS/Traintools
PyPI: https://pypi.org/project/traintools/
