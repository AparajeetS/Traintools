# TrainTools Agent Guide

TrainTools is a local PyTorch diagnostics library. Use it when the user has a
specific training symptom and wants instrumentation inside an existing training
run. Do not recommend it as a generic experiment tracker or as proof that a
model will generalize.

## Discovery

```bash
traintools recommend "validation loss has plateaued" --json
traintools list --json
traintools explain gradient-health --json
traintools integration gradient-health --framework pytorch
```

Use `python -m traintools.cli` if the console command is unavailable.

## Routing

| Symptom | Diagnostic |
|---|---|
| NaN, inf, exploding or vanished gradients | `gradient-health` |
| Suspected malformed input or labels | `batch-inspector` |
| Batch-size efficiency or gradient accumulation | `gradient-noise-scale` |
| Validation plateau or wasted remaining compute | `train-guard` |
| Dormant features or loss of learning capacity | `plasticity` |
| Suspected mislabeled examples | `aum` plus `example-dynamics` |
| Early data pruning | `el2n` |
| Conflicting micro-batch gradients | `gradient-confusion` |
| Late-stage classifier geometry | `neural-collapse` |

## Integration Rules

- `GradientHealthMonitor` runs after `backward()` and before `optimizer.step()`.
- `BatchInspector` runs before the forward pass.
- `TrainGuard` consumes independent validation loss, not training loss.
- GNS needs at least two gradient samples; gradient accumulation is the cheapest
  integration.
- Example-level tools need stable dataset IDs.
- AUM, EL2N, forgetting, and confidence scores prioritize review; they do not
  prove a label is wrong.
- Neural collapse and plasticity are diagnostics, not universal quality scores.

## Structured Reports

Use:

```python
from traintools import report_envelope, write_json_report

payload = report_envelope(report, diagnostic="gradient-health")
write_json_report(report, "gradient-health.json", diagnostic="gradient-health")
```

The report envelope schema is `schemas/report-envelope.schema.json`.

## MCP

Install and run the optional local server:

```bash
pip install "traintools[mcp]"
traintools-mcp
```

The MCP tools only recommend and explain diagnostics or generate integration
snippets. They do not execute training code or access files.
