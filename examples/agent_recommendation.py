"""Ask TrainTools which diagnostic fits a concrete training symptom."""

from __future__ import annotations

import json

from traintools import integration_snippet, recommend_diagnostics


problem = "My PyTorch loss becomes NaN and gradients explode after a few hundred steps"
recommendations = recommend_diagnostics(problem, framework="pytorch")
print(json.dumps(recommendations, indent=2))

if recommendations:
    print("\nSuggested integration:\n")
    print(integration_snippet(recommendations[0]["id"], "pytorch"))
