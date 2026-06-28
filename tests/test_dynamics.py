import torch

from traintools.dynamics import ExampleDynamicsTracker, DynamicsSummary


def _logits_for(preds, targets, confidence=5.0):
    logits = torch.zeros(len(preds), 3)
    for i, pred in enumerate(preds):
        logits[i, pred] = confidence
        logits[i, targets[i]] += 0.5
    return logits


def test_example_dynamics_tracks_forgetting_events():
    tracker = ExampleDynamicsTracker(min_observations=1)
    ids = torch.tensor([10, 11, 12])
    y = torch.tensor([0, 1, 2])

    tracker.update(ids, _logits_for([0, 1, 2], y), y, step=0)
    summary = tracker.update(ids, _logits_for([0, 2, 2], y), y, step=1)

    assert isinstance(summary, DynamicsSummary)
    assert tracker.examples[11].forgetting_events == 1
    assert tracker.examples[10].forgetting_events == 0
    assert summary.total_forgetting_events == 1


def test_example_dynamics_unforgettable_and_never_learned():
    tracker = ExampleDynamicsTracker(min_observations=1)
    ids = [1, 2, 3]
    y = torch.tensor([0, 1, 2])

    tracker.update(ids, _logits_for([0, 0, 0], y), y, step=0)
    tracker.update(ids, _logits_for([0, 0, 0], y), y, step=1)

    unforgettable_ids = {ex.example_id for ex in tracker.unforgettable()}
    never_ids = {ex.example_id for ex in tracker.never_learned()}
    assert 1 in unforgettable_ids
    assert 2 in never_ids
    assert 3 in never_ids


def test_example_dynamics_cartography_regions():
    tracker = ExampleDynamicsTracker(
        easy_confidence=0.7,
        hard_confidence=0.4,
        ambiguous_variability=0.1,
        min_observations=3,
    )
    ids = [1, 2, 3]
    y = torch.tensor([0, 1, 2])

    tracker.update(ids, _logits_for([0, 1, 0], y, confidence=6.0), y, step=0)
    tracker.update(ids, _logits_for([0, 2, 2], y, confidence=6.0), y, step=1)
    tracker.update(ids, _logits_for([0, 1, 0], y, confidence=6.0), y, step=2)

    easy = {ex.example_id for ex in tracker.cartography_region("easy")}
    ambiguous = {ex.example_id for ex in tracker.cartography_region("ambiguous")}
    hard = {ex.example_id for ex in tracker.cartography_region("hard")}

    assert 1 in easy
    assert 2 in ambiguous
    assert 3 in hard


def test_example_dynamics_validates_shapes():
    tracker = ExampleDynamicsTracker()
    logits = torch.randn(2, 3)
    y = torch.tensor([0, 1])

    try:
        tracker.update([1], logits, y)
    except ValueError as exc:
        assert "example_ids length" in str(exc)
    else:
        raise AssertionError("expected ValueError")
