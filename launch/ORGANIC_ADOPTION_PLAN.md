# TrainTools Organic Adoption Plan

The goal is ten real users who run a diagnostic and report what happened, not a
large number of low-intent impressions.

## Release Week

1. Publish version 0.6.0 with the CLI, schemas, problem guides, and MCP server.
2. Create a GitHub release showing three commands and one honest limitation.
3. Publish one focused article: "My PyTorch loss became NaN: a diagnostic loop,
   not a checklist."
4. Post the batch-size guide where users already discuss GNS or gradient
   accumulation.
5. Ask three researchers or engineers to try one diagnostic on an existing run.

## Useful Channels

- PyTorch Forums: answer existing questions with a minimal reproducible example.
- Hugging Face Forums: demonstrate the callback on a real Trainer run.
- GitHub issues in training repositories: contribute a diagnosis only where it
  directly resolves the issue.
- Reddit ML communities: one technical case study, with results and limitations.
- Hacker News or Show HN: only after the CLI and documentation are polished.
- Papers with Code or research-code directories where the underlying methods fit.

Do not cross-post identical marketing text, purchase stars, manufacture users,
or claim benchmark wins that were not measured.

## Conversion

Every article should lead to one command:

```bash
pip install traintools
traintools recommend "describe your training problem"
```

Every issue template should ask:

- framework and version;
- diagnostic used;
- integration snippet;
- structured JSON report;
- expected and observed behavior;
- whether the signal changed a training decision.

## Evidence To Publish Next

- known-corruption label-audit benchmark;
- deliberately broken gradient benchmark;
- GNS recommendation versus batch-size sweep;
- TrainGuard compute saved versus missed improvement;
- runtime and memory overhead;
- false alarms and failure cases.

These benchmarks will do more for recommendations than broad claims about the
number of included diagnostics.
