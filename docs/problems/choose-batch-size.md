# Choose A PyTorch Batch Size

Use Gradient Noise Scale when the question is whether the current batch is
below or above the optimization-efficiency frontier.

```bash
pip install traintools
traintools integration gradient-noise-scale --framework pytorch
```

Gradient accumulation is the preferred integration because its micro-batch
gradients are already being computed:

```python
from traintools import GradientAccumulationGNS

gns = GradientAccumulationGNS(model, micro_batch_size=micro_batch_size)
for inputs, targets in micro_batches:
    (loss_fn(model(inputs), targets) / accumulation_steps).backward()
    gns.record_microbatch()

result = gns.compute(step=step)
```

Interpret `critical_batch` as an estimated optimization scale. Benchmark actual
throughput and memory before changing the batch. GNS does not predict final
generalization and may vary during training.
