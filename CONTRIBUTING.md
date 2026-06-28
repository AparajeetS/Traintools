# Contributing

Thanks for helping make `traintools` useful.

## Development Setup

```bash
git clone https://github.com/AparajeetS/Traintools.git
cd Traintools
pip install -e ".[dev]"
python -m pytest
```

## Good Contributions

- Reproducible bug reports with model shape, batch shape, PyTorch version, and a short traceback.
- Tests for numerical edge cases.
- Benchmarks from real training runs.
- Small diagnostics that answer one practical training question.

## Design Principles

- Diagnostics should not mutate model state unless that behavior is explicit.
- Core tools should work with PyTorch only.
- Optional integrations belong behind extras such as `[hf]` or `[full]`.
- Thresholds should be configurable and documented as heuristics.
