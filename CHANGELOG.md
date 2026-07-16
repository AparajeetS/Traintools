# Changelog

## 0.6.2 - 2026-07-17

- Preserve the case-sensitive GitHub namespace required by MCP Registry OIDC.

## 0.6.1 - 2026-07-17

- Add official MCP Registry metadata and PyPI ownership verification.
- Add `traintools mcp` for a stable registry-compatible stdio launch command.

## 0.6.0 - 2026-07-16

- Add a machine-readable capability registry and symptom-based recommender.
- Add the `traintools` discovery and integration CLI with JSON output.
- Add stable JSON report envelopes and schemas.
- Add `AGENTS.md`, `llms.txt`, problem-oriented guides, and agent examples.
- Add an optional local MCP server using the official Python MCP SDK.
- Add an organic-adoption and evidence plan without manufactured engagement.

## 0.5.1 - 2026-06-28

- Fix the primary README install command to use `pip install traintools`.
- Add an explicit GitHub source link near the top of the README.

## 0.5.0 - 2026-06-28

- Add `AUMTracker` for Area Under the Margin mislabeled-example ranking.
- Add `EL2NTracker` and `el2n_scores` for early example-importance scoring.
- Add `NeuralCollapseMonitor` for NC1/NC2/NC3/NCC geometry diagnostics.

## 0.4.0 - 2026-06-28

- Add `ExampleDynamicsTracker` for example forgetting, never-learned examples,
  unforgettable examples, and dataset-cartography-style regions.
- Add `GradientConfusionMonitor` and `gradient_confusion_from_grads` for
  pairwise micro-batch gradient conflict diagnostics.
- Document the new paper-backed diagnostics and cite their source papers.

## 0.3.0 - 2026-06-28

- Add `BatchInspector` for batch/data health checks.
- Add `GradientHealthMonitor` for gradient, clipping, and update-ratio diagnostics.
- Export the new diagnostics from the top-level `traintools` package.
- Improve public package metadata with project URLs and author metadata.
- Rework README toward a broader ML diagnostics toolkit.

## 0.2.0 - 2026-05-24

- Add free GNS during gradient accumulation.
- Fix the GNS estimator to use Bessel-corrected variance and bias-corrected signal.
- Add HuggingFace callback GNS support.
- Improve plasticity diagnostics.

## 0.1.0 - 2026-05-24

- Initial PyPI release.
