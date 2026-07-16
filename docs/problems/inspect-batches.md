# Inspect Broken Batches And Labels

Use `BatchInspector` before the forward pass to detect empty tensors, NaNs,
infinities, extreme scales, constant tensors, class imbalance, and invalid
classification labels.

```python
from traintools import BatchInspector

inspector = BatchInspector(expected_num_classes=10, max_abs_value=1e4)
report = inspector.inspect(inputs, targets, step=step)
if not report.ok:
    print(report)
```

Run it on startup and periodically after data augmentations. A clean report
only describes inspected batches; it does not validate the entire dataset.
