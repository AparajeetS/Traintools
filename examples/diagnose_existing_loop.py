"""Minimal TrainTools diagnostics inside an existing PyTorch loop.

This example is intentionally boring: no framework wrapper, no tracker service,
and no extra training abstraction. Copy the relevant blocks into the loop that
already exists.
"""

from __future__ import annotations

import torch
from torch import nn

from traintools import BatchInspector, ExampleDynamicsTracker, GradientHealthMonitor


def train_one_epoch(model, dataloader, optimizer, *, step0: int = 0):
    loss_fn = nn.CrossEntropyLoss()
    batch_inspector = BatchInspector(expected_num_classes=10, max_abs_value=1e4)
    grad_health = GradientHealthMonitor(max_grad_norm=1.0)
    dynamics = ExampleDynamicsTracker()

    model.train()
    for offset, batch in enumerate(dataloader):
        step = step0 + offset

        # Prefer stable dataset ids from the dataset. Fall back to row positions.
        if len(batch) == 3:
            example_ids, inputs, targets = batch
        else:
            inputs, targets = batch
            example_ids = torch.arange(len(targets)) + step * len(targets)

        batch_report = batch_inspector.inspect(inputs, targets, step=step)
        if not batch_report.ok:
            print(batch_report)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()

        grad_report = grad_health.inspect(
            model,
            step=step,
            lr=optimizer.param_groups[0]["lr"],
        )
        if not grad_report.ok:
            print(grad_report)

        optimizer.step()

        dynamics.update(
            example_ids=example_ids,
            logits=logits.detach(),
            targets=targets,
            step=step,
        )

    return dynamics.summary()
