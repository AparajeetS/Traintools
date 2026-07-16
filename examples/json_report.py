"""Write a diagnostic result in the stable TrainTools report envelope."""

from __future__ import annotations

import torch
import torch.nn as nn

from traintools import GradientHealthMonitor, write_json_report


model = nn.Linear(4, 2)
loss = model(torch.randn(8, 4)).square().mean()
loss.backward()
report = GradientHealthMonitor().inspect(model, step=1, lr=1e-3)
path = write_json_report(
    report,
    "gradient-health-report.json",
    diagnostic="gradient-health",
)
print(path)
