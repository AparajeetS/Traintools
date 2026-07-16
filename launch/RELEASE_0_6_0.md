# TrainTools 0.6.0: Diagnostics That Humans And Agents Can Route Correctly

TrainTools now has a machine-readable diagnostic registry and a CLI that maps a
specific training problem to relevant tools:

```bash
pip install traintools
traintools recommend "validation loss plateaued" --json
traintools integration train-guard --framework huggingface
```

The release also adds structured JSON reports, focused problem guides,
`AGENTS.md`, `llms.txt`, and an optional local MCP server:

```bash
pip install "traintools[mcp]"
traintools-mcp
```

The recommender is transparent keyword routing, not an AI diagnosis. Each result
includes requirements, call timing, citations, and limitations, and it abstains
when no supported diagnostic matches.
