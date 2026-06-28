import torch

from traintools.aum import AUMTracker
from traintools.collapse import NeuralCollapseMonitor
from traintools.importance import EL2NTracker, el2n_scores


def test_aum_tracker_ranks_low_margin_examples():
    tracker = AUMTracker(min_observations=2)
    ids = [1, 2]
    y = torch.tensor([0, 1])
    good_bad = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    tracker.update(ids, good_bad, y, step=0)
    tracker.update(ids, good_bad, y, step=1)

    lowest = tracker.lowest_aum(1)[0]

    assert lowest.example_id == 2
    assert lowest.aum < 0


def test_aum_suspicious_threshold():
    tracker = AUMTracker(low_aum_threshold=0.0, min_observations=1)
    logits = torch.tensor([[0.0, 3.0], [3.0, 0.0]])
    y = torch.tensor([0, 0])
    tracker.update([10, 11], logits, y)

    suspicious = {ex.example_id for ex in tracker.suspicious()}

    assert 10 in suspicious
    assert 11 not in suspicious


def test_el2n_scores_higher_for_wrong_confident_prediction():
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    y = torch.tensor([0, 0])
    scores = el2n_scores(logits, y)

    assert scores[1] > scores[0]


def test_el2n_tracker_orders_examples():
    tracker = EL2NTracker(min_observations=1)
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    y = torch.tensor([0, 0])
    tracker.update([1, 2], logits, y)

    assert tracker.highest(1)[0].example_id == 2
    assert tracker.lowest(1)[0].example_id == 1


def test_neural_collapse_monitor_detects_tight_class_clusters():
    torch.manual_seed(0)
    means = torch.tensor([[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]])
    features = torch.cat([m + 0.01 * torch.randn(8, 2) for m in means], dim=0)
    labels = torch.tensor([0] * 8 + [1] * 8 + [2] * 8)

    result = NeuralCollapseMonitor(nc1_warn=0.1).measure(features, labels, classifier_weight=means)

    assert result.n_classes == 3
    assert result.nc1_within_to_between < 0.01
    assert result.nc2_etf_deviation < 0.05
    assert result.ncc_accuracy == 1.0
    assert result.nc3_classifier_alignment is not None


def test_neural_collapse_monitor_validates_inputs():
    monitor = NeuralCollapseMonitor()
    try:
        monitor.measure(torch.randn(4, 2), torch.zeros(4, dtype=torch.long))
    except ValueError as exc:
        assert "at least two classes" in str(exc)
    else:
        raise AssertionError("expected ValueError")
