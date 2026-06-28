import torch
import torch.nn as nn

from traintools.gradhealth import GradientHealthMonitor, GradientHealthResult


def test_gradient_health_clean_backward():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(6, 4)
    y = torch.randint(0, 2, (6,))

    loss_fn(model(x), y).backward()
    result = GradientHealthMonitor(exploding_threshold=1e6).inspect(model, step=3, lr=1e-3)

    assert isinstance(result, GradientHealthResult)
    assert result.total_grad_norm > 0
    assert result.total_param_norm > 0
    assert result.global_update_ratio is not None
    assert result.layers


def test_gradient_health_detects_missing_gradients():
    model = nn.Linear(4, 2)

    result = GradientHealthMonitor().inspect(model)

    assert not result.ok
    assert any("no gradients" in warning for warning in result.warnings)


def test_gradient_health_detects_nonfinite_gradient():
    model = nn.Linear(2, 1)
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    model.weight.grad[0, 0] = float("nan")

    result = GradientHealthMonitor().inspect(model)

    assert not result.ok
    assert any("non-finite" in warning for warning in result.warnings)


def test_gradient_health_clip_coefficient():
    model = nn.Linear(2, 1)
    for param in model.parameters():
        param.grad = torch.ones_like(param) * 10.0

    result = GradientHealthMonitor(max_grad_norm=1.0).inspect(model)

    assert result.clip_coef is not None
    assert result.clip_coef < 1.0
    assert any("clipping threshold" in warning for warning in result.warnings)


def test_gradient_health_update_ratio_warning():
    model = nn.Linear(2, 1)
    for param in model.parameters():
        param.grad = torch.ones_like(param) * 100.0

    result = GradientHealthMonitor(max_update_ratio=1e-4).inspect(model, lr=1.0)

    assert not result.ok
    assert any("update/weight ratio" in warning for warning in result.warnings)
