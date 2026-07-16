# TrainTools 0.6.1: Official MCP Registry Support

TrainTools can now be discovered by MCP-compatible agents and launched as a
local stdio server without a persistent installation:

```bash
uvx --with mcp traintools mcp
```

This patch release adds the official MCP Registry manifest, PyPI ownership
verification, and a stable `traintools mcp` command. The server remains local,
read-only, and focused on recommending and explaining training diagnostics.
