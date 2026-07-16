# Find Examples Worth Auditing

TrainTools provides three complementary rankings:

- `AUMTracker`: persistently poor true-class margins;
- `ExampleDynamicsTracker`: forgetting, confidence, and variability;
- `EL2NTracker`: early example difficulty and pruning priority.

```python
from traintools import AUMTracker, ExampleDynamicsTracker, EL2NTracker

aum = AUMTracker()
dynamics = ExampleDynamicsTracker()
el2n = EL2NTracker()

logits = model(inputs)
aum.update(example_ids, logits, labels, step=step)
dynamics.update(example_ids, logits, labels, step=step)
el2n.update(example_ids, logits, labels, step=step)
```

Use stable example IDs. Compare rankings and inspect the original data before
removing anything. A hard, rare, or distribution-edge example can look
suspicious while being correctly labeled.
