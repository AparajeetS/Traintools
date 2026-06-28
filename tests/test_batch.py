import torch

from traintools.batch import BatchInspector, BatchReport


def test_batch_inspector_clean_batch():
    inspector = BatchInspector(expected_num_classes=3)
    x = torch.randn(8, 4)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    report = inspector.inspect(x, y, step=7)

    assert isinstance(report, BatchReport)
    assert report.ok
    assert report.batch_size == 8
    assert report.label_distribution == {0: 3, 1: 3, 2: 2}


def test_batch_inspector_detects_nonfinite_and_scale():
    inspector = BatchInspector(max_abs_value=100.0)
    x = torch.tensor([[1.0, float("nan")], [float("inf"), 1000.0]])

    report = inspector.inspect(x, step=1)

    assert not report.ok
    text = "\n".join(report.warnings)
    assert "non-finite" in text
    assert "exceeds" in text


def test_batch_inspector_nested_inputs_and_bad_labels():
    inspector = BatchInspector(expected_num_classes=3)
    inputs = {"image": torch.randn(4, 3, 8, 8), "meta": [torch.ones(4, 2)]}
    targets = torch.tensor([0, 1, 7, 7])

    report = inspector.inspect(inputs, targets)

    assert report.batch_size == 4
    assert len(report.tensors) == 3
    assert not report.ok
    assert any("labels outside" in warning for warning in report.warnings)


def test_batch_inspector_detects_imbalance():
    inspector = BatchInspector(class_imbalance_warn=0.75)
    y = torch.tensor([1, 1, 1, 1, 0])

    report = inspector.inspect(torch.randn(5, 2), y)

    assert report.imbalance_ratio == 0.8
    assert any("dominant class" in warning for warning in report.warnings)
